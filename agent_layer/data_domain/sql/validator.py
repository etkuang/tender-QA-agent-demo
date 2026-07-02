# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.data_domain.schemas import (
    DataContextBundle,
    SQLCandidate,
    SQLValidationIssue,
    SQLValidationResult,
)


class SQLPolicyValidator:
    forbidden_function_names = {
        "LOAD_FILE",
        "PG_READ_FILE",
        "PG_READ_BINARY_FILE",
        "LO_IMPORT",
        "LO_EXPORT",
        "READ_CSV",
        "READ_JSON",
        "READ_PARQUET",
    }

    def __init__(self, dialect: str, max_rows: int):
        try:
            import sqlglot
            from sqlglot import exp
        except ImportError as exc:
            raise RuntimeError("The data-domain SQL workflow requires sqlglot.") from exc
        self.sqlglot = sqlglot
        self.exp = exp
        self.dialect = dialect
        self.max_rows = max_rows

    def validate(self, candidate: SQLCandidate, context: DataContextBundle) -> SQLValidationResult:
        issues = []
        try:
            expressions = self.sqlglot.parse(candidate.statement, read=context.dialect)
        except Exception as exc:
            return self._invalid(candidate.statement, "syntax_error", exc.__class__.__name__)
        if len(expressions) != 1:
            return self._invalid(candidate.statement, "multiple_statements", "Only one SQL statement is allowed.")

        tree = expressions[0]
        if not isinstance(tree, (self.exp.Select, self.exp.Union)):
            return self._invalid(candidate.statement, "not_read_only", "Only SELECT, WITH, or UNION queries are allowed.")

        forbidden_types = tuple(
            expression_type
            for name in [
                "Insert",
                "Update",
                "Delete",
                "Create",
                "Drop",
                "Alter",
                "Command",
                "Copy",
                "Merge",
                "Transaction",
                "Use",
            ]
            if (expression_type := getattr(self.exp, name, None)) is not None
        )
        if forbidden_types and any(tree.find_all(*forbidden_types)):
            issues.append(SQLValidationIssue(code="forbidden_statement", message="Write, DDL, system, or transaction statements are forbidden."))

        if not context.access_policy.allow_select_star and list(tree.find_all(self.exp.Star)):
            issues.append(SQLValidationIssue(code="select_star", message="SELECT * is not allowed."))

        cte_names = {cte.alias_or_name for cte in tree.find_all(self.exp.CTE)}
        tables = sorted(
            {
                table.name
                for table in tree.find_all(self.exp.Table)
                if table.name not in cte_names
            }
        )
        if not tables:
            issues.append(SQLValidationIssue(code="missing_table", message="The query must read at least one approved table."))
        unknown_tables = [table for table in tables if table not in context.table_names]
        if unknown_tables:
            issues.append(SQLValidationIssue(code="unknown_table", message=f"Unknown or unauthorized tables: {', '.join(unknown_tables)}."))

        columns = sorted({column.name for column in tree.find_all(self.exp.Column) if column.name != "*"})
        unknown_columns = [column for column in columns if column not in context.column_names]
        if unknown_columns:
            issues.append(SQLValidationIssue(code="unknown_column", message=f"Unknown or unauthorized columns: {', '.join(unknown_columns)}."))
        denied_columns = [column for column in columns if column in context.sensitive_column_names]
        if denied_columns:
            issues.append(SQLValidationIssue(code="sensitive_column", message=f"Sensitive columns are not allowed: {', '.join(denied_columns)}."))

        for function in tree.find_all(self.exp.Anonymous):
            if function.name.upper() in self.forbidden_function_names:
                issues.append(SQLValidationIssue(code="forbidden_function", message=f"Forbidden function: {function.name}."))

        limit, limit_issue = self._enforce_limit(tree, context)
        if limit_issue is not None:
            issues.append(limit_issue)

        statement = tree.sql(dialect=context.dialect)
        return SQLValidationResult(
            valid=not issues,
            statement=statement,
            tables=tables,
            columns=columns,
            limit=limit,
            issues=issues,
        )

    def _enforce_limit(self, tree, context: DataContextBundle):
        if not context.access_policy.require_limit:
            return None, None
        limit_expression = tree.args.get("limit")
        if limit_expression is None:
            tree.set("limit", self.exp.Limit(expression=self.exp.Literal.number(context.access_policy.max_rows)))
            return context.access_policy.max_rows, None
        value_expression = limit_expression.expression
        if not isinstance(value_expression, self.exp.Literal) or not value_expression.is_int:
            return None, SQLValidationIssue(code="invalid_limit", message="LIMIT must be a positive integer literal.")
        limit_value = value_expression.to_py()
        if not isinstance(limit_value, int) or limit_value <= 0:
            return None, SQLValidationIssue(code="invalid_limit", message="LIMIT must be positive.")
        if limit_value > context.access_policy.max_rows:
            limit_expression.set("expression", self.exp.Literal.number(context.access_policy.max_rows))
            return context.access_policy.max_rows, None
        return limit_value, None

    @staticmethod
    def _invalid(statement: str, code: str, message: str) -> SQLValidationResult:
        return SQLValidationResult(
            valid=False,
            statement=statement,
            issues=[SQLValidationIssue(code=code, message=message)],
        )
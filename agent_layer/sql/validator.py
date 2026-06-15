# coding: utf-8

from agent_layer.errors import SQLValidationError
from agent_layer.sql.schemas import SQLValidationResult, ViewSchema


class SqlglotValidator:
    """AST validator for one read-only SELECT/CTE statement."""

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

    def __init__(self, dialect: str, max_rows: int, max_joins: int, max_subqueries: int):
        try:
            import sqlglot
            from sqlglot import exp
        except ImportError as exc:
            raise RuntimeError("The target SQL workflow requires the sqlglot package.") from exc
        self.sqlglot = sqlglot
        self.exp = exp
        self.dialect = dialect
        self.max_rows = max_rows
        self.max_joins = max_joins
        self.max_subqueries = max_subqueries

    def validate(self, statement: str, view_schemas: list[ViewSchema]) -> SQLValidationResult:
        try:
            expressions = self.sqlglot.parse(statement, read=self.dialect)
        except Exception as exc:
            raise SQLValidationError from exc
        if len(expressions) != 1:
            raise SQLValidationError
        tree = expressions[0]
        if not isinstance(tree, (self.exp.Select, self.exp.Union)):
            raise SQLValidationError

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
            raise SQLValidationError

        schema_map = {schema.name: schema for schema in view_schemas}
        cte_names = {cte.alias_or_name for cte in tree.find_all(self.exp.CTE)}
        tables = sorted(
            {
                table.name
                for table in tree.find_all(self.exp.Table)
                if table.name not in cte_names
            }
        )
        if not tables or any(table not in schema_map for table in tables):
            raise SQLValidationError

        columns = sorted({column.name for column in tree.find_all(self.exp.Column) if column.name != "*"})
        allowed_columns = set()
        sensitive_columns = set()
        for table in tables:
            allowed_columns.update(schema_map[table].columns)
            sensitive_columns.update(schema_map[table].sensitive_columns)
        if any(column not in allowed_columns for column in columns):
            raise SQLValidationError
        if any(column in sensitive_columns for column in columns):
            raise SQLValidationError

        if sum(1 for _ in tree.find_all(self.exp.Join)) > self.max_joins:
            raise SQLValidationError
        if sum(1 for _ in tree.find_all(self.exp.Subquery)) > self.max_subqueries:
            raise SQLValidationError

        for function in tree.find_all(self.exp.Anonymous):
            if function.name.upper() in self.forbidden_function_names:
                raise SQLValidationError

        limit = self._enforce_limit(tree)
        return SQLValidationResult(
            statement=tree.sql(dialect=self.dialect),
            tables=tables,
            columns=columns,
            limit=limit,
        )

    def _enforce_limit(self, tree) -> int:
        limit_expression = tree.args.get("limit")
        if limit_expression is None:
            tree.set("limit", self.exp.Limit(expression=self.exp.Literal.number(self.max_rows)))
            return self.max_rows
        value_expression = limit_expression.expression
        if not isinstance(value_expression, self.exp.Literal) or not value_expression.is_int:
            raise SQLValidationError
        value = int(value_expression.this)
        if value <= 0:
            raise SQLValidationError
        if value > self.max_rows:
            limit_expression.set("expression", self.exp.Literal.number(self.max_rows))
            return self.max_rows
        return value

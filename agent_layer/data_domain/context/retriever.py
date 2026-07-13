# coding: utf-8
# @Author: Wang Qingkang

from agent_layer.config import Settings
from agent_layer.schemas import Category, DependencyOutcome
from agent_layer.data_domain.context.catalog import DataContextCatalog
from agent_layer.data_domain.schemas import (
    ApprovedSQLExample,
    BusinessGlossaryTerm,
    DataAccessPolicy,
    DataContextBundle,
    DataMetricDefinition,
    DataTableContext,
)


class DataContextRetriever:
    def __init__(self, catalog: DataContextCatalog, settings: Settings):
        self.catalog = catalog
        self.settings = settings

    def retrieve(
        self,
        query: str,
        category: Category,
    ) -> DataContextBundle:
        text = query.lower()
        candidate_tables = self.catalog.tables_for_domain(category)
        selected_tables = self._rank_tables(text, candidate_tables)
        table_names = {table.name for table in selected_tables}
        metrics = self._rank_metrics(text, self.catalog.metrics_for_domain(category))
        glossary_terms = self._rank_glossary(text, self.catalog.glossary_for_domain(category))
        examples = self._rank_examples(text, self.catalog.examples_for_domain(category), table_names)
        relationships = self.catalog.relationships_for_tables(table_names)
        denied_columns = sorted(
            {
                column.name
                for table in selected_tables
                for column in table.columns
                if column.is_sensitive
            }
        )
        return DataContextBundle(
            domain=category,
            dialect=self.settings.sql_dialect,
            tables=selected_tables,
            relationships=relationships,
            metrics=metrics,
            glossary_terms=glossary_terms,
            examples=examples,
            access_policy=DataAccessPolicy(
                allowed_tables=[table.name for table in selected_tables],
                denied_columns=denied_columns,
                max_rows=self.settings.sql_max_rows,
            ),
        )

    def _rank_tables(
        self,
        text: str,
        tables: list[DataTableContext],
    ) -> list[DataTableContext]:
        scored = []
        for table in tables:
            score = self._contains_score(text, table.name)
            score += self._contains_score(text, table.description)
            for column in table.columns:
                score += self._contains_score(text, column.name)
                score += self._contains_score(text, column.description)
            scored.append((score, table))
        ordered = [
            table
            for score, table in sorted(scored, key=lambda item: item[0], reverse=True)
            if score > 0
        ]
        if not ordered:
            ordered = tables
        return ordered[: self.settings.sql_context_table_limit]

    def _rank_metrics(
        self,
        text: str,
        metrics: list[DataMetricDefinition],
    ) -> list[DataMetricDefinition]:
        selected = []
        for metric in metrics:
            if any(
                self._contains_score(text, term)
                for term in [metric.name, metric.description]
            ):
                selected.append(metric)

    def _rank_glossary(
        self,
        text: str,
        glossary_terms: list[BusinessGlossaryTerm],
    ) -> list[BusinessGlossaryTerm]:
        selected = []
        for term in glossary_terms:
            terms = [term.term, term.definition, *term.synonyms]
            if any(self._contains_score(text, value) for value in terms):
                selected.append(term)
        return selected

    def _rank_examples(
        self,
        text: str,
        examples: list[ApprovedSQLExample],
        table_names: set[str],
    ) -> list[ApprovedSQLExample]:
        scored = []
        for example in examples:
            score = self._contains_score(text, example.question)
            if table_names.intersection(example.tables):
                score += 2
            scored.append((score, example))
        ordered = [
            example
            for score, example in sorted(scored, key=lambda item: item[0], reverse=True)
            if score > 0
        ]
        return ordered[: self.settings.sql_context_example_limit]

    @staticmethod
    def _contains_score(text: str, value: str | None) -> int:
        if not value:
            return 0
        lowered = value.lower()
        if lowered in text:
            return 2
        return sum(1 for token in lowered.split() if token and token in text)
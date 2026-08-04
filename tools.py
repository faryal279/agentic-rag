import sqlite3
from langchain_core.tools import tool
from duckduckgo_search import DDGS


def make_document_search_tool(retriever):
    if retriever is None:
        return None

    @tool
    def search_documents(query: str) -> str:
        """Search the user's uploaded documents for information relevant to the query."""
        # Fixed: Updated method for latest LangChain retriever
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant content found in the uploaded documents."
        formatted = []
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_label = f" (page {page + 1})" if page is not None else ""
            formatted.append(f"[Source {i}{page_label}]: {doc.page_content[:600]}")
        return "\n\n".join(formatted)

    return search_documents


@tool
def search_web(query: str) -> str:
    """Search the live web for current or general information."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No web results found."
        formatted = []
        for i, r in enumerate(results, start=1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            formatted.append(f"[{i}] {title}\n{body}\nSource: {href}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Web search failed: {e}"


def make_sql_tool(db_path: str):
    SCHEMA_DESCRIPTION = """\
Available tables:
sales(id, product, category, units_sold, revenue, month)
papers(id, title, field, year, citations)
"""

    @tool
    def query_database(sql_query: str) -> str:
        """Run a read-only SQL SELECT query against the demo database."""
        cleaned = sql_query.strip().rstrip(";")
        if not cleaned.lower().startswith("select"):
            return "Only SELECT queries are allowed."
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute(cleaned)
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            conn.close()
            if not rows:
                return "Query ran successfully but returned no rows."
            header = " | ".join(columns)
            lines = [header, "-" * len(header)]
            for row in rows:
                lines.append(" | ".join(str(v) for v in row))
            return "\n\n".join(lines)
        except Exception as e:
            return f"SQL error: {e}. {SCHEMA_DESCRIPTION}"

    return query_database

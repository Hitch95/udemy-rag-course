"""
SQL RAG Agent for Tesla Motors Database
Using create_agent with custom SQL tools (compatible with Ollama)
"""

import os
import re
from typing import List, Dict, Any

from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent

# ============================================================================
# CONFIGURATION
# ============================================================================
DB_PATH = "6_sql_rag/tesla_motors_data.db"
MODEL_NAME = "qwen3:1.7b"
TEMPERATURE = 0
MAX_TOKENS = 1000

# ============================================================================
# STEP 1: DATABASE SETUP
# ============================================================================
print("Setting up Tesla Motors database...")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# Check connection and get basic info
try:
    tables = db.get_usable_table_names()
    print(f"✓ Database connected successfully")
    print(f"✓ Found {len(tables)} tables: {', '.join(tables)}")
except Exception as e:
    print(f"✗ Database connection failed: {e}")
    exit(1)

# Get schema information
SCHEMA = db.get_table_info()
print("✓ Retrieved database schema")

# ============================================================================
# STEP 2: MODEL SETUP
# ============================================================================
print("Initializing Ollama model...")

llm = ChatOllama(
    model=MODEL_NAME,
    base_url="http://localhost:11434",
    temperature=TEMPERATURE
)

print(f"✓ Initialized Ollama chat model ({MODEL_NAME})")

# ============================================================================
# STEP 3: SQL TOOLS - One for each step
# ============================================================================

@tool
def get_database_schema(table_name: str = None) -> str:
    """Get database schema information for SQL query generation.
    Use this first to understand table structure before creating queries."""
    print(f"🔍 Getting schema for: {table_name if table_name else 'all tables'}")
    
    if table_name:
        try:
            tables = db.get_usable_table_names()
            if table_name.lower() in [t.lower() for t in tables]:
                result = db.get_table_info([table_name])
                print(f"✓ Retrieved schema for table: {table_name}")
                return result
            else:
                return f"Error: Table '{table_name}' not found. Available tables: {', '.join(tables)}"
        except Exception as e:
            return f"Error getting table info: {e}"
    else:
        print("✓ Retrieved full database schema")
        return SCHEMA


@tool
def generate_sql_query(question: str, schema_info: str = None) -> str:
    """Generate a SQL SELECT query from a natural language question using database schema.
    Always use this after getting schema information."""
    print(f"Generating SQL for: {question[:100]}...")
    
    schema_to_use = schema_info if schema_info else SCHEMA
    
    prompt = f"""
Based on this database schema:
{schema_to_use}

Generate a SQL query to answer this question: {question}

Rules:
- Use only SELECT statements
- Include only existing columns and tables
- Add appropriate WHERE, GROUP BY, ORDER BY clauses as needed
- Limit results to 10 rows unless specified otherwise
- Use proper SQL syntax for SQLite

Return only the SQL query, nothing else.
"""
    
    try:
        response = llm.invoke(prompt)
        query = response.content.strip()
        print(f"✓ Generated SQL query")
        return query
    except Exception as e:
        return f"Error generating query: {e}"


@tool
def validate_sql_query(query: str) -> str:
    """Validate SQL query for safety and syntax before execution.
    Returns 'Valid: <query>' if safe or 'Error: <message>' if unsafe."""
    print(f"Validating SQL: {query[:100]}...")
    
    query = query.strip()
    
    # Remove common SQL formatting
    clean_query = re.sub(r'```sql\s*', '', query, flags=re.IGNORECASE)
    clean_query = re.sub(r'```\s*', '', clean_query)
    clean_query = clean_query.strip()
    
    # Block multiple statements
    if clean_query.count(";") > 1 or (clean_query.endswith(";") and ";" in clean_query[:-1]):
        return "Error: Multiple statements not allowed"
    
    clean_query = clean_query.rstrip(";").strip()
    
    # Must be SELECT only
    if not clean_query.lower().startswith("select"):
        return "Error: Only SELECT statements allowed"
    
    # Block dangerous operations
    dangerous_patterns = [
        r'\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b',
        r'\b(EXEC|EXECUTE)\b',
        r'--',  # SQL comments
        r'/\*',  # Block comments
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, clean_query, re.IGNORECASE):
            return f"Error: Unsafe SQL pattern detected"
    
    try:
        if clean_query.lower().count('select') > 1 and 'from' not in clean_query.lower():
            return "Error: Multiple SELECT statements not allowed"
        
        if clean_query.count('(') != clean_query.count(')'):
            return "Error: Unbalanced parentheses"
        
        print("✓ Query validation passed")
        return f"Valid: {clean_query}"
        
    except Exception as e:
        return f"Error: Syntax validation failed: {e}"


@tool
def execute_sql_query(query: str) -> str:
    """Execute a validated SQL query and return results.
    Only use this after validating the query for safety."""
    print(f"Executing SQL: {query[:100]}...")
    
    try:
        clean_query = query.strip()
        if clean_query.startswith("Valid: "):
            clean_query = clean_query[7:]
        
        clean_query = re.sub(r'```sql\s*', '', clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r'```\s*', '', clean_query)
        clean_query = clean_query.strip().rstrip(";")
        
        result = db.run(clean_query)
        print("✓ Query executed successfully")
        
        if result:
            return f"Query Results:\n{result}"
        else:
            return "Query executed successfully but returned no results."
            
    except Exception as e:
        error_msg = f"Execution Error: {str(e)}"
        print(f"✗ {error_msg}")
        return error_msg


@tool
def fix_sql_error(original_query: str, error_message: str, question: str) -> str:
    """Fix a failed SQL query by analyzing the error and generating a corrected version.
    Use this when validation or execution fails."""
    print(f"🔧 Fixing SQL error: {error_message[:100]}...")
    
    fix_prompt = f"""
The following SQL query failed:
Query: {original_query}
Error: {error_message}
Original Question: {question}

Database Schema:
{SCHEMA}

Analyze the error and provide a corrected SQL query that:
1. Fixes the specific error mentioned
2. Still answers the original question
3. Uses only valid table and column names from the schema
4. Follows SQLite syntax rules

Return only the corrected SQL query, nothing else.
"""
    
    try:
        response = llm.invoke(fix_prompt)
        fixed_query = response.content.strip()
        print("✓ Generated fixed SQL query")
        return fixed_query
    except Exception as e:
        return f"Error generating fix: {e}"


@tool
def analyze_query_results(question: str, query: str, results: str) -> str:
    """Convert SQL query results into a natural language answer.
    Use this as the final step to provide a user-friendly response."""
    print("📊 Analyzing results and generating answer...")
    
    analysis_prompt = f"""
Original Question: {question}
SQL Query Used: {query}
Query Results: {results}

Provide a clear, natural language answer to the original question based on the query results.
Be specific and include relevant numbers/data from the results.
If the results are empty or unclear, mention that as well.
"""
    
    try:
        response = llm.invoke(analysis_prompt)
        answer = response.content.strip()
        print("✓ Generated natural language answer")
        return answer
    except Exception as e:
        return f"Error analyzing results: {e}"


print("✓ Created all SQL workflow tools")

# ============================================================================
# STEP 4: SYSTEM PROMPT
# ============================================================================

SQL_SYSTEM_PROMPT = f"""You are an expert SQL analyst working with a Tesla Motors database.

Database Schema:
{SCHEMA}

Your workflow for answering questions:
1. Use `get_database_schema` first to understand available tables and columns (if needed)
2. Use `generate_sql_query` to create SQL based on the question
3. Use `validate_sql_query` to check the query for safety and syntax
4. Use `execute_sql_query` to run the validated query
5. If there's an error, use `fix_sql_error` to correct it and try again (up to 3 times)
6. Use `analyze_query_results` to provide a natural language answer

Rules:
- Always follow the workflow step by step
- If a query fails, use the fix tool and try again
- Provide clear, informative answers
- Be precise with table and column names
- Handle errors gracefully and try to fix them
- If you fail after 3 attempts, explain what went wrong

Available tools for each step:
- get_database_schema: Get table structure info
- generate_sql_query: Create SQL from question
- validate_sql_query: Check query safety/syntax  
- execute_sql_query: Run the query
- fix_sql_error: Fix failed queries
- analyze_query_results: Generate natural language answer

Remember: Always validate queries before executing them for safety.
"""

# ============================================================================
# STEP 5: CREATE AGENT
# ============================================================================

tools = [
    get_database_schema,
    generate_sql_query,
    validate_sql_query, 
    execute_sql_query,
    fix_sql_error,
    analyze_query_results
]

# Create the SQL agent using create_agent (signature: model, tools, system_prompt=...)
sql_agent = create_agent(
    llm, 
    tools, 
    system_prompt=SQL_SYSTEM_PROMPT
)

print("✓ Created SQL agent with create_agent")

# ============================================================================
# STEP 6: QUERY FUNCTION
# ============================================================================

def ask_sql(question: str):
    """Ask the SQL agent a question using the full workflow."""
    print(f"\n{'='*60}")
    print(f"SQL AGENT - Question: {question}")
    print('='*60)
    
    for event in sql_agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values"
    ):
        msg = event["messages"][-1]
        
        # Show tool usage
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                print(f"\n🔧 Using: {tc['name']}")
                args_str = str(tc['args'])
                if len(args_str) > 200:
                    args_str = args_str[:200] + "..."
                print(f"Args: {args_str}")
        
        # Show final answer
        elif hasattr(msg, 'content') and msg.content:
            print(f"\n💬 Answer:\n{msg.content}")

# ============================================================================
# STEP 7: MAIN - Interactive Chat
# ============================================================================

def main():
    print("\nTesla Motors SQL Agent - Type 'quit' to exit")
    print("Ask questions about Tesla vehicles in the database")
    print("\nExample questions:")
    print("- What's the average price of Tesla Model S?")
    print("- How many vehicles are in the database?")
    print("- Which model has the highest battery capacity?")
    print("- Compare the specifications of Model S and Model X")
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the Tesla Motors SQL Agent. Goodbye!")
            break
        if question:
            ask_sql(question)


if __name__ == "__main__":
    main()

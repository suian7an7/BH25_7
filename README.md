1.Please enter your own API key in the .env file.

2.When running, please use gpt_database_qa_ver5.py.

3.The AI agent will automatically retrieve tables from the database, so please ignore other data-fetching scripts in the folder.

4.Since the AI agent performs a lot of reasoning, I’ve added semantic_mapping_chembl_35.txt to help it think more efficiently. You can also add more content to it yourself.

5.The current version (ver5) allows increasing the degree of contextual association—just modify the value of K. However, please be aware of the token limit.

6.Version 5 can only read data from SQLite databases. You can add support for other databases yourself.

7.You can customize the UI output—either concise or detailed—but again, keep the token limit in mind.

CREATE TABLE conversation_sessions (
	id INTEGER NOT NULL, 
	session_id VARCHAR(100) NOT NULL, 
	user_id VARCHAR(100) NOT NULL, 
	title VARCHAR(200), 
	status VARCHAR(20) NOT NULL, 
	created_at DATETIME NOT NULL, 
	last_activity_at DATETIME NOT NULL, 
	expires_at DATETIME, 
	max_turns INTEGER NOT NULL, 
	context_window INTEGER NOT NULL, 
	session_metadata JSON, 
	PRIMARY KEY (id)
);
CREATE UNIQUE INDEX ix_conversation_sessions_session_id ON conversation_sessions (session_id);
CREATE INDEX ix_conversation_sessions_user_id ON conversation_sessions (user_id);
CREATE TABLE query_logs (
	id INTEGER NOT NULL, 
	query_text TEXT NOT NULL, 
	query_type VARCHAR(12) NOT NULL, 
	query_hash VARCHAR(64), 
	user_id VARCHAR(100) NOT NULL, 
	session_id VARCHAR(100), 
	user_agent VARCHAR(500), 
	ip_address VARCHAR(45), 
	status VARCHAR(9) NOT NULL, 
	error_message TEXT, 
	response_time_ms FLOAT, 
	processing_time_ms FLOAT, 
	embedding_time_ms FLOAT, 
	search_time_ms FLOAT, 
	llm_time_ms FLOAT, 
	prompt_tokens INTEGER, 
	completion_tokens INTEGER, 
	total_tokens INTEGER, 
	search_limit INTEGER, 
	search_offset INTEGER, 
	search_filter JSON, 
	results_count INTEGER, 
	max_similarity_score FLOAT, 
	min_similarity_score FLOAT, 
	avg_similarity_score FLOAT, 
	model_used VARCHAR(100), 
	llm_provider VARCHAR(50), 
	finish_reason VARCHAR(50), 
	request_metadata JSON, 
	response_metadata JSON, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id)
);
CREATE INDEX ix_query_logs_query_hash ON query_logs (query_hash);
CREATE INDEX ix_query_logs_user_id ON query_logs (user_id);
CREATE TABLE conversation_turns (
	id INTEGER NOT NULL, 
	session_id INTEGER NOT NULL, 
	turn_number INTEGER NOT NULL, 
	user_query TEXT NOT NULL, 
	assistant_response TEXT NOT NULL, 
	response_time_ms INTEGER, 
	token_count INTEGER, 
	turn_metadata JSON, 
	created_at DATETIME NOT NULL, 
	PRIMARY KEY (id), 
	FOREIGN KEY(session_id) REFERENCES conversation_sessions (id)
);

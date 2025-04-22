-- Create the keyspace
CREATE KEYSPACE IF NOT EXISTS rag_keyspace
WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

-- Use the keyspace
USE rag_keyspace;

-- Create tables

CREATE TABLE IF NOT EXISTS users_new (
    user_id UUID PRIMARY KEY,
    username TEXT,
    email TEXT,
    his_job TEXT,
    password TEXT
);

CREATE TABLE IF NOT EXISTS session_table (
    session_id UUID PRIMARY KEY,
    title TEXT,
    timestamp TIMESTAMP,
    user_id UUID
);

CREATE TABLE IF NOT EXISTS response_table (
    partition_id UUID PRIMARY KEY,
    question TEXT,
    answer TEXT,
    timestamp TIMESTAMP,
    evaluation BOOLEAN
);

CREATE TABLE IF NOT EXISTS response_session (
    partition_id UUID PRIMARY KEY,
    session_id UUID,
    table_response_id UUID
);

CREATE TABLE IF NOT EXISTS vectores (
    partition_id UUID PRIMARY KEY,
    document_text TEXT,
    document_content BLOB,
    vector BLOB
);

from services.cassandra_connector import CassandraConnector

def test_connection():
    print("Attempting to connect to Cassandra...")
    try:
        with CassandraConnector() as db:
            print("✓ Connection successful!")
            print("Cluster metadata:", db.cluster.metadata)
            print("Keyspaces:", list(db.cluster.metadata.keyspaces.keys()))
    except Exception as e:
        print("✗ Connection failed:", str(e))

if __name__ == "__main__":
    test_connection()
import psycopg2
import logging
from datetime import datetime

class DatabaseHandler:
    """
    Handler for database operations with PostgreSQL for the trading bot.
    Manages trade records in the database.
    """
    
    def __init__(self, dbname, user, password, host="localhost", port="5432"):
        """
        Initialize the database handler with connection parameters.
        
        Args:
            dbname: Database name
            user: Database user
            password: Database password
            host: Database host (default: localhost)
            port: Database port (default: 5432)
        """
        self.connection_params = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port
        }
        self.logger = logging.getLogger("DatabaseHandler")
        self.connection = None
        self.cursor = None
        
    def connect(self):
        """
        Connect to the PostgreSQL database.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.connection = psycopg2.connect(**self.connection_params)
            self.cursor = self.connection.cursor()
            self.logger.info(f"Successfully connected to database {self.connection_params['dbname']}")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            return False
            
    def disconnect(self):
        """
        Close the database connection.
        """
        try:
            if self.cursor:
                self.cursor.close()
            if self.connection:
                self.connection.close()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
    
    def create_trades_table(self):
        """
        Create the trades table if it doesn't already exist.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create table if it doesn't exist
            create_table_query = """
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                order_id VARCHAR(255),
                symbol VARCHAR(50) NOT NULL,
                entry_time TIMESTAMP NOT NULL,
                entry_price NUMERIC(16, 8) NOT NULL,
                size NUMERIC(16, 8) NOT NULL,
                stop_loss NUMERIC(16, 8) NOT NULL,
                take_profit NUMERIC(16, 8) NOT NULL,
                exit_time TIMESTAMP,
                exit_price NUMERIC(16, 8),
                profit_loss NUMERIC(16, 8),
                result VARCHAR(10),
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            self.cursor.execute(create_table_query)
            self.connection.commit()
            self.logger.info("Trades table created or already exists")
            return True
        except Exception as e:
            self.logger.error(f"Error creating trades table: {e}")
            self.connection.rollback()
            return False
    
    def insert_trade(self, order_id, symbol, entry_time, entry_price, size, stop_loss, take_profit):
        """
        Insert a new active trade into the database.
        
        Args:
            order_id: Order ID from trading platform
            symbol: Trading symbol
            entry_time: Time of trade entry
            entry_price: Entry price
            size: Trade size/amount
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            int: ID of the inserted record or None if failed
        """
        try:
            # Insert the new trade
            insert_query = """
            INSERT INTO trades 
            (order_id, symbol, entry_time, entry_price, size, stop_loss, take_profit, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, TRUE)
            RETURNING id;
            """
            
            self.cursor.execute(
                insert_query, 
                (
                    order_id, 
                    symbol, 
                    entry_time, 
                    entry_price, 
                    size, 
                    stop_loss, 
                    take_profit
                )
            )
            
            trade_id = self.cursor.fetchone()[0]
            self.connection.commit()
            self.logger.info(f"Trade inserted with ID: {trade_id}")
            return trade_id
        except Exception as e:
            self.logger.error(f"Error inserting trade: {e}")
            self.connection.rollback()
            return None
    
    def update_trade_result(self, order_id, exit_time, exit_price, profit_loss, result):
        """
        Update a trade with exit information when it's closed.
        
        Args:
            order_id: Order ID to identify the trade
            exit_time: Time of trade exit
            exit_price: Exit price
            profit_loss: Profit/loss amount
            result: Result of the trade ('Win' or 'Loss')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Update the trade with exit information
            update_query = """
            UPDATE trades
            SET exit_time = %s, exit_price = %s, profit_loss = %s, result = %s, is_active = FALSE
            WHERE order_id = %s AND is_active = TRUE;
            """
            
            self.cursor.execute(
                update_query, 
                (
                    exit_time,
                    exit_price,
                    profit_loss,
                    result,
                    order_id
                )
            )
            
            rows_updated = self.cursor.rowcount
            self.connection.commit()
            
            if rows_updated == 0:
                self.logger.warning(f"No active trade found with order_id: {order_id}")
                return False
                
            self.logger.info(f"Trade with order_id {order_id} updated with result: {result}")
            return True
        except Exception as e:
            self.logger.error(f"Error updating trade: {e}")
            self.connection.rollback()
            return False
    
    def get_active_trades(self, symbol=None):
        """
        Get all active trades, optionally filtered by symbol.
        
        Args:
            symbol: Trading symbol to filter by (optional)
            
        Returns:
            list: List of active trades as dictionaries
        """
        try:
            if symbol:
                # Get active trades for a specific symbol
                query = "SELECT * FROM trades WHERE is_active = TRUE AND symbol = %s;"
                self.cursor.execute(query, (symbol,))
            else:
                # Get all active trades
                query = "SELECT * FROM trades WHERE is_active = TRUE;"
                self.cursor.execute(query)
                
            columns = [desc[0] for desc in self.cursor.description]
            trades = []
            
            for row in self.cursor.fetchall():
                trade = dict(zip(columns, row))
                trades.append(trade)
                
            self.logger.info(f"Retrieved {len(trades)} active trades")
            return trades
        except Exception as e:
            self.logger.error(f"Error retrieving active trades: {e}")
            return []
    
    def get_trade_history(self, symbol=None, limit=100):
        """
        Get trade history, optionally filtered by symbol, with a limit.
        
        Args:
            symbol: Trading symbol to filter by (optional)
            limit: Maximum number of trades to retrieve (default: 100)
            
        Returns:
            list: List of trades as dictionaries
        """
        try:
            if symbol:
                # Get trade history for a specific symbol
                query = """
                SELECT * FROM trades 
                WHERE symbol = %s
                ORDER BY entry_time DESC 
                LIMIT %s;
                """
                self.cursor.execute(query, (symbol, limit))
            else:
                # Get all trade history
                query = """
                SELECT * FROM trades 
                ORDER BY entry_time DESC 
                LIMIT %s;
                """
                self.cursor.execute(query, (limit,))
                
            columns = [desc[0] for desc in self.cursor.description]
            trades = []
            
            for row in self.cursor.fetchall():
                trade = dict(zip(columns, row))
                trades.append(trade)
                
            self.logger.info(f"Retrieved {len(trades)} historical trades")
            return trades
        except Exception as e:
            self.logger.error(f"Error retrieving trade history: {e}")
            return []
            
    def get_performance_metrics(self, symbol=None):
        """
        Get performance metrics from trade history.
        
        Args:
            symbol: Trading symbol to filter by (optional)
            
        Returns:
            dict: Performance metrics
        """
        try:
            where_clause = "WHERE symbol = %s" if symbol else ""
            params = (symbol,) if symbol else ()
            
            query = f"""
            SELECT 
                COUNT(*) AS total_trades,
                SUM(CASE WHEN result = 'Win' THEN 1 ELSE 0 END) AS win_count,
                SUM(CASE WHEN result = 'Loss' THEN 1 ELSE 0 END) AS loss_count,
                SUM(profit_loss) AS total_profit_loss,
                MAX(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) AS largest_win,
                MIN(CASE WHEN profit_loss < 0 THEN profit_loss ELSE 0 END) AS largest_loss
            FROM trades
            {where_clause};
            """
            
            self.cursor.execute(query, params)
            metrics = dict(zip([desc[0] for desc in self.cursor.description], self.cursor.fetchone()))
            
            # Calculate additional metrics
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = metrics['win_count'] / metrics['total_trades']
            else:
                metrics['win_rate'] = 0
                
            self.logger.info(f"Retrieved performance metrics: {metrics}")
            return metrics
        except Exception as e:
            self.logger.error(f"Error retrieving performance metrics: {e}")
            return {
                'total_trades': 0,
                'win_count': 0,
                'loss_count': 0,
                'total_profit_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'win_rate': 0
            }
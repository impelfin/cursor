# core.py
import asyncio

class FastMCP:
    def __init__(self, name, dependencies=None, timeout=None):
        self.name = name
        self.dependencies = dependencies or []
        self.timeout = timeout
        self.tools = {}

    def tool(self):
        def decorator(func):
            # Register your tool logic
            self.tools[func.__name__] = func
            async def wrapper(*args, **kwargs):
                if self.timeout:
                    try:
                        return await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout)
                    except asyncio.TimeoutError:
                        return f"MCP error -32001: Request timed out (exceeded {self.timeout}s)"
                    except Exception as e:
                        return f"MCP error -32002: {str(e)}"
                else:
                    return await func(*args, **kwargs)
            return wrapper
        return decorator

    def resource(self, path):
        def decorator(func):
            # Register your resource logic
            return func
        return decorator

    def prompt(self):
        def decorator(func):
            # Register your prompt logic
            return func
        return decorator

class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}
    
    def get_content(self) -> str:
        return self.content
    
    def get_metadata(self) -> dict:
        return self.metadata
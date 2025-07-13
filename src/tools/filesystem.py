import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

from tools.base import BaseTool
from tools.schemas import ToolParameter, ToolResult, ToolSchema


class ReadFileTool(BaseTool):
    """Tool to read file contents."""
    
    @property
    def name(self) -> str:
        return "read_file"
    
    @property
    def description(self) -> str:
        return "Read the contents of a file"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string", 
                    description="Path to the file to read",
                    required=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"File does not exist: {path}"
                )
            
            if not file_path.is_file():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path is not a file: {path}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ToolResult(
                success=True,
                content=content,
                metadata={"file_size": len(content), "path": str(file_path)}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Error reading file: {str(e)}"
            )


class WriteFileTool(BaseTool):
    """Tool to write content to a file."""
    
    @property
    def name(self) -> str:
        return "write_file"
    
    @property
    def description(self) -> str:
        return "Write content to a file (creates or overwrites)"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to write",
                    required=True
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write to the file",
                    required=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        content = kwargs.get("content", "")
        
        try:
            file_path = Path(path)
            
            # Create parent directories if they don't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResult(
                success=True,
                content=f"Successfully wrote {len(content)} characters to {path}",
                metadata={"bytes_written": len(content), "path": str(file_path)}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Error writing file: {str(e)}"
            )


class EditFileTool(BaseTool):
    """Tool to edit existing files with find/replace operations."""
    
    @property
    def name(self) -> str:
        return "edit_file"
    
    @property
    def description(self) -> str:
        return "Edit a file by replacing text patterns"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to edit",
                    required=True
                ),
                ToolParameter(
                    name="find_text",
                    type="string",
                    description="Text to find and replace",
                    required=True
                ),
                ToolParameter(
                    name="replace_text",
                    type="string",
                    description="Text to replace with",
                    required=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        find_text = kwargs.get("find_text")
        replace_text = kwargs.get("replace_text")
        
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"File does not exist: {path}"
                )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if find_text not in content:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Text to replace not found in file: {find_text}"
                )
            
            new_content = content.replace(find_text, replace_text)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            replacements = content.count(find_text)
            return ToolResult(
                success=True,
                content=f"Successfully replaced {replacements} occurrence(s) in {path}",
                metadata={"replacements": replacements, "path": str(file_path)}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Error editing file: {str(e)}"
            )


class CreateDirectoryTool(BaseTool):
    """Tool to create directories."""
    
    @property
    def name(self) -> str:
        return "create_directory"
    
    @property
    def description(self) -> str:
        return "Create a directory (and parent directories if needed)"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the directory to create",
                    required=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        
        try:
            dir_path = Path(path)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return ToolResult(
                success=True,
                content=f"Successfully created directory: {path}",
                metadata={"path": str(dir_path)}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Error creating directory: {str(e)}"
            )


class ListDirectoryTool(BaseTool):
    """Tool to list directory contents."""
    
    @property
    def name(self) -> str:
        return "list_directory"
    
    @property
    def description(self) -> str:
        return "List files and directories in a given path"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the directory to list",
                    required=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        
        try:
            dir_path = Path(path)
            if not dir_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Directory does not exist: {path}"
                )
            
            if not dir_path.is_dir():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path is not a directory: {path}"
                )
            
            items = []
            for item in dir_path.iterdir():
                item_type = "directory" if item.is_dir() else "file"
                size = item.stat().st_size if item.is_file() else None
                items.append({
                    "name": item.name,
                    "type": item_type,
                    "size": size,
                    "path": str(item)
                })
            
            content = "\n".join([
                f"{item['name']} ({item['type']}" + 
                (f", {item['size']} bytes)" if item['size'] is not None else ")")
                for item in items
            ])
            
            return ToolResult(
                success=True,
                content=content,
                metadata={"items": items, "count": len(items)}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Error listing directory: {str(e)}"
            )


class DeleteFileTool(BaseTool):
    """Tool to delete files or directories."""
    
    @property
    def name(self) -> str:
        return "delete_file"
    
    @property
    def description(self) -> str:
        return "Delete a file or directory"
    
    def get_schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file or directory to delete",
                    required=True
                )
            ]
        )
    
    def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path")
        
        try:
            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Path does not exist: {path}"
                )
            
            if file_path.is_file():
                file_path.unlink()
                item_type = "file"
            elif file_path.is_dir():
                shutil.rmtree(file_path)
                item_type = "directory"
            else:
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Cannot delete: {path}"
                )
            
            return ToolResult(
                success=True,
                content=f"Successfully deleted {item_type}: {path}",
                metadata={"path": str(file_path), "type": item_type}
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Error deleting: {str(e)}"
            )
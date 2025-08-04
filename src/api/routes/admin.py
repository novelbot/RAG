"""
Admin API routes for user management and system administration
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel
import asyncio

from ...auth.dependencies import get_current_user, get_admin_user

# Alias for clarity
require_admin = get_admin_user
from ...auth.schemas import UserCreate, UserResponse, UserUpdate
from ...auth.user_manager import UserManager
from ...core.database import get_db
from ...core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Create a new user (admin only)
    """
    try:
        user_manager = UserManager(db)
        
        # Check if user already exists
        if user_manager.get_user_by_username(user_data.username):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this username already exists"
            )
        
        if user_manager.get_user_by_email(user_data.email):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User with this email already exists"
            )
        
        # Create the user
        new_user = user_manager.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            role=user_data.role,
            department=getattr(user_data, 'department', None)
        )
        
        logger.info(f"User {user_data.username} created by admin {current_user.username}")
        
        return UserResponse(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            full_name=getattr(new_user, 'full_name', None),
            role=new_user.role,
            department=getattr(new_user, 'department', None),
            is_active=new_user.is_active,
            timezone=getattr(new_user, 'timezone', 'UTC'),
            bio=getattr(new_user, 'bio', None),
            avatar_url=getattr(new_user, 'avatar_url', None),
            is_superuser=getattr(new_user, 'is_superuser', False),
            is_verified=getattr(new_user, 'is_verified', True),
            created_at=new_user.created_at,
            updated_at=getattr(new_user, 'updated_at', new_user.created_at)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    List all users (admin only)
    """
    try:
        user_manager = UserManager(db)
        users = user_manager.get_all_users(limit=limit, offset=offset)
        
        return [
            UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                full_name=getattr(user, 'full_name', None),
                role=user.role,
                department=getattr(user, 'department', None),
                is_active=user.is_active,
                timezone=getattr(user, 'timezone', 'UTC'),
                bio=getattr(user, 'bio', None),
                avatar_url=getattr(user, 'avatar_url', None),
                is_superuser=getattr(user, 'is_superuser', False),
                is_verified=getattr(user, 'is_verified', True),
                created_at=user.created_at,
                updated_at=getattr(user, 'updated_at', user.created_at)
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Update a user (admin only)
    """
    try:
        user_manager = UserManager(db)
        
        # Get the user to update
        user = user_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update user
        updated_user = user_manager.update_user(user_id, **user_update.model_dump(exclude_unset=True))
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info(f"User {user.username} updated by admin {current_user.username}")
        
        return UserResponse(
            id=updated_user.id,
            username=updated_user.username,
            email=updated_user.email,
            full_name=getattr(updated_user, 'full_name', None),
            role=updated_user.role,
            department=getattr(updated_user, 'department', None),
            is_active=updated_user.is_active,
            timezone=getattr(updated_user, 'timezone', 'UTC'),
            bio=getattr(updated_user, 'bio', None),
            avatar_url=getattr(updated_user, 'avatar_url', None),
            is_superuser=getattr(updated_user, 'is_superuser', False),
            is_verified=getattr(updated_user, 'is_verified', True),
            created_at=updated_user.created_at,
            updated_at=getattr(updated_user, 'updated_at', updated_user.created_at)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Delete a user (admin only)
    """
    try:
        user_manager = UserManager(db)
        
        # Get the user to delete
        user = user_manager.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prevent self-deletion
        if user.id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete your own account"
            )
        
        # Delete user
        success = user_manager.delete_user(user_id)
        
        if success:
            logger.info(f"User {user.username} deleted by admin {current_user.username}")
            return {"message": "User deleted successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(require_admin)
):
    """
    Get user details (admin only)
    """
    try:
        user_manager = UserManager(db)
        user = user_manager.get_user_by_id(user_id)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=getattr(user, 'full_name', None),
            role=user.role,
            department=getattr(user, 'department', None),
            is_active=user.is_active,
            timezone=getattr(user, 'timezone', 'UTC'),
            bio=getattr(user, 'bio', None),
            avatar_url=getattr(user, 'avatar_url', None),
            is_superuser=getattr(user, 'is_superuser', False),
            is_verified=getattr(user, 'is_verified', True),
            created_at=user.created_at,
            updated_at=getattr(user, 'updated_at', user.created_at)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


# Pydantic models for episode processing
class RetryNovelRequest(BaseModel):
    novel_ids: List[int]
    conservative: bool = True
    max_retries: int = 5

class RetryNovelResponse(BaseModel):
    status: str
    message: str
    task_id: str = None
    success_count: int = 0
    failed_novels: List[int] = []

class ProcessAllEpisodesRequest(BaseModel):
    force_reprocess: bool = False
    conservative: bool = True
    batch_size: int = 5
    drop_existing_collection: bool = False

class ProcessAllEpisodesResponse(BaseModel):
    status: str
    message: str
    task_id: str = None
    total_novels: int = 0
    estimated_episodes: int = 0

# Global task storage (in production, use Redis or database)
_background_tasks = {}

async def process_all_episodes_background(task_id: str, force_reprocess: bool = False, conservative: bool = True, batch_size: int = 5, drop_existing_collection: bool = False):
    """Background task for processing all episodes"""
    try:
        from ...core.config import get_config
        from ...database.base import DatabaseManager
        from ...embedding.manager import EmbeddingManager
        from ...milvus.client import MilvusClient
        from ...episode.manager import EpisodeRAGManager, EpisodeRAGConfig
        from sqlalchemy import text
        
        _background_tasks[task_id] = {"status": "initializing", "progress": 0, "total": 0}
        
        config = get_config()
        
        # Initialize dependencies
        db_manager = DatabaseManager(config.database)
        
        if config.embedding_providers:
            provider_configs = list(config.embedding_providers.values())
        else:
            provider_configs = [config.embedding]
            
        embedding_manager = EmbeddingManager(provider_configs)
        milvus_client = MilvusClient(config.milvus)
        
        # Get all novels with episodes
        with db_manager.get_connection() as conn:
            result = conn.execute(text('SELECT DISTINCT novel_id FROM episode ORDER BY novel_id'))
            all_novel_ids = [row[0] for row in result.fetchall()]
            
            # Get total episode count for progress tracking
            result = conn.execute(text('SELECT COUNT(*) FROM episode'))
            total_episodes = result.fetchone()[0]
        
        _background_tasks[task_id] = {
            "status": "running", 
            "progress": 0, 
            "total": len(all_novel_ids),
            "total_episodes": total_episodes,
            "processed_episodes": 0
        }
        
        # Episode processing configuration
        episode_config = EpisodeRAGConfig(
            processing_batch_size=batch_size,
            vector_dimension=1024
        )
        
        episode_manager = EpisodeRAGManager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config
        )
        
        # Connect to Milvus
        milvus_client.connect()
        
        # Setup collection (drop if requested)
        await episode_manager.setup_collection(drop_existing=drop_existing_collection)
        
        success_count = 0
        failed_novels = []
        total_processed_episodes = 0
        
        for i, novel_id in enumerate(all_novel_ids):
            try:
                # Update progress
                _background_tasks[task_id]["progress"] = i
                _background_tasks[task_id]["current_novel"] = novel_id
                
                # Process novel
                result = await episode_manager.process_novel(novel_id, force_reprocess=force_reprocess)
                success_count += 1
                episodes_processed = result.get("episodes_processed", 0)
                total_processed_episodes += episodes_processed
                
                _background_tasks[task_id]["processed_episodes"] = total_processed_episodes
                
                # Wait between novels
                wait_time = 10 if conservative else 5
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Novel {novel_id} processing failed: {e}")
                failed_novels.append(novel_id)
                await asyncio.sleep(15)  # Wait longer after failure
        
        # Update final status
        _background_tasks[task_id] = {
            "status": "completed",
            "progress": len(all_novel_ids),
            "total": len(all_novel_ids),
            "success_count": success_count,
            "failed_novels": failed_novels,
            "total_episodes": total_episodes,
            "processed_episodes": total_processed_episodes,
            "completed_at": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        _background_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": asyncio.get_event_loop().time()
        }

async def process_novels_background(task_id: str, novel_ids: List[int], conservative: bool = True):
    """Background task for processing novels"""
    try:
        from ...core.config import get_config
        from ...database.base import DatabaseManager
        from ...embedding.manager import EmbeddingManager
        from ...milvus.client import MilvusClient
        from ...episode.manager import EpisodeRAGManager, EpisodeRAGConfig
        
        _background_tasks[task_id] = {"status": "running", "progress": 0, "total": len(novel_ids)}
        
        config = get_config()
        
        # Initialize dependencies
        db_manager = DatabaseManager(config.database)
        
        if config.embedding_providers:
            provider_configs = list(config.embedding_providers.values())
        else:
            provider_configs = [config.embedding]
            
        embedding_manager = EmbeddingManager(provider_configs)
        milvus_client = MilvusClient(config.milvus)
        
        # Conservative settings
        batch_size = 2 if conservative else 5
        episode_config = EpisodeRAGConfig(
            processing_batch_size=batch_size,
            vector_dimension=1024
        )
        
        episode_manager = EpisodeRAGManager(
            database_manager=db_manager,
            embedding_manager=embedding_manager,
            milvus_client=milvus_client,
            config=episode_config
        )
        
        # Connect to Milvus
        milvus_client.connect()
        
        success_count = 0
        failed_novels = []
        
        for i, novel_id in enumerate(novel_ids):
            try:
                # Update progress
                _background_tasks[task_id]["progress"] = i
                _background_tasks[task_id]["current_novel"] = novel_id
                
                # Process novel
                await episode_manager.process_novel(novel_id)
                success_count += 1
                
                # Wait between novels
                wait_time = 10 if conservative else 5
                await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Novel {novel_id} processing failed: {e}")
                failed_novels.append(novel_id)
                await asyncio.sleep(15)  # Wait longer after failure
        
        # Update final status
        _background_tasks[task_id] = {
            "status": "completed",
            "progress": len(novel_ids),
            "total": len(novel_ids),
            "success_count": success_count,
            "failed_novels": failed_novels,
            "completed_at": asyncio.get_event_loop().time()
        }
        
    except Exception as e:
        _background_tasks[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": asyncio.get_event_loop().time()
        }


@router.post("/episodes/retry-novels", response_model=RetryNovelResponse)
async def retry_failed_novels_api(
    request: RetryNovelRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_admin)
):
    """
    Retry processing for specific failed novels (admin only)
    
    Uses improved stability settings to retry novels that failed
    during previous processing runs. This is a background task.
    """
    try:
        if not request.novel_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one novel ID is required"
            )
        
        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Start background task
        background_tasks.add_task(
            process_novels_background,
            task_id,
            request.novel_ids,
            request.conservative
        )
        
        logger.info(f"Started novel retry task {task_id} for novels {request.novel_ids} by admin {current_user.username}")
        
        return RetryNovelResponse(
            status="started",
            message=f"Started retry processing for {len(request.novel_ids)} novels",
            task_id=task_id
        )
        
    except Exception as e:
        logger.error(f"Error starting novel retry: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start novel retry processing"
        )


@router.post("/episodes/process-all", response_model=ProcessAllEpisodesResponse)
async def process_all_episodes_api(
    request: ProcessAllEpisodesRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_admin)
):
    """
    Process all episodes from database (admin only)
    
    Initializes Milvus collection and processes all episodes from RDB.
    This is equivalent to the CLI command: data ingest --database --episode-mode
    """
    try:
        # Get episode count estimate
        from ...core.config import get_config
        from ...database.base import DatabaseManager
        from sqlalchemy import text
        
        config = get_config()
        db_manager = DatabaseManager(config.database)
        
        with db_manager.get_connection() as conn:
            # Get novel count
            result = conn.execute(text('SELECT COUNT(DISTINCT novel_id) FROM episode'))
            total_novels = result.fetchone()[0]
            
            # Get episode count
            result = conn.execute(text('SELECT COUNT(*) FROM episode'))
            estimated_episodes = result.fetchone()[0]
        
        db_manager.close()
        
        # Generate task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Start background task
        background_tasks.add_task(
            process_all_episodes_background,
            task_id,
            request.force_reprocess,
            request.conservative,
            request.batch_size,
            request.drop_existing_collection
        )
        
        logger.info(f"Started full episode processing task {task_id} by admin {current_user.username}")
        
        return ProcessAllEpisodesResponse(
            status="started",
            message=f"Started processing all episodes from database ({total_novels} novels, ~{estimated_episodes} episodes)",
            task_id=task_id,
            total_novels=total_novels,
            estimated_episodes=estimated_episodes
        )
        
    except Exception as e:
        logger.error(f"Error starting full episode processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start full episode processing"
        )


@router.get("/episodes/retry-status/{task_id}")
async def get_retry_status(
    task_id: str,
    current_user = Depends(require_admin)
):
    """
    Get status of a retry processing task
    """
    if task_id not in _background_tasks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    task_info = _background_tasks[task_id]
    
    return {
        "task_id": task_id,
        "status": task_info.get("status"),
        "progress": task_info.get("progress", 0),
        "total": task_info.get("total", 0),
        "current_novel": task_info.get("current_novel"),
        "success_count": task_info.get("success_count", 0),
        "failed_novels": task_info.get("failed_novels", []),
        "total_episodes": task_info.get("total_episodes", 0),
        "processed_episodes": task_info.get("processed_episodes", 0),
        "error": task_info.get("error"),
        "completed_at": task_info.get("completed_at"),
        "failed_at": task_info.get("failed_at")
    }
"""
Admin API routes for user management and system administration
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

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
            role=new_user.role,
            department=getattr(new_user, 'department', None),
            created_at=new_user.created_at,
            updated_at=getattr(new_user, 'updated_at', new_user.created_at),
            is_active=new_user.is_active,
            is_superuser=getattr(new_user, 'is_superuser', False),
            is_verified=getattr(new_user, 'is_verified', True)
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
                role=user.role,
                department=getattr(user, 'department', None),
                created_at=user.created_at,
                updated_at=getattr(user, 'updated_at', user.created_at),
                is_active=user.is_active,
                is_superuser=getattr(user, 'is_superuser', False),
                is_verified=getattr(user, 'is_verified', True)
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
        updated_user = user_manager.update_user(user_id, **user_update.dict(exclude_unset=True))
        
        logger.info(f"User {user.username} updated by admin {current_user.username}")
        
        return UserResponse(
            id=updated_user.id,
            username=updated_user.username,
            email=updated_user.email,
            role=updated_user.role,
            department=getattr(updated_user, 'department', None),
            created_at=updated_user.created_at,
            updated_at=getattr(updated_user, 'updated_at', updated_user.created_at),
            is_active=updated_user.is_active,
            is_superuser=getattr(updated_user, 'is_superuser', False),
            is_verified=getattr(updated_user, 'is_verified', True)
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
            role=user.role,
            department=getattr(user, 'department', None),
            created_at=user.created_at,
            updated_at=getattr(user, 'updated_at', user.created_at),
            is_active=user.is_active,
            is_superuser=getattr(user, 'is_superuser', False),
            is_verified=getattr(user, 'is_verified', True)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )
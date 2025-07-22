"""
Unit tests for Milvus RBAC module.
"""
import pytest
from unittest.mock import Mock
from datetime import datetime

from src.milvus.rbac import (
    RBACManager, UserContext, Permission, AccessScope, AccessRule,
    create_rbac_manager
)
from src.core.exceptions import RBACError, PermissionError


class TestPermission:
    """Test Permission enum."""
    
    def test_permission_values(self):
        """Test Permission enum values."""
        assert Permission.READ.value == "read"
        assert Permission.WRITE.value == "write"
        assert Permission.DELETE.value == "delete"
        assert Permission.ADMIN.value == "admin"


class TestAccessScope:
    """Test AccessScope enum."""
    
    def test_access_scope_values(self):
        """Test AccessScope enum values."""
        assert AccessScope.COLLECTION.value == "collection"
        assert AccessScope.DOCUMENT.value == "document"
        assert AccessScope.USER.value == "user"
        assert AccessScope.GROUP.value == "group"


class TestAccessRule:
    """Test AccessRule dataclass."""
    
    def test_access_rule_creation(self):
        """Test AccessRule creation."""
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="doc123",
            conditions={"user_id": "user1"}
        )
        
        assert rule.scope == AccessScope.DOCUMENT
        assert rule.permission == Permission.READ
        assert rule.resource_id == "doc123"
        assert rule.conditions == {"user_id": "user1"}

    def test_access_rule_matches_exact(self):
        """Test access rule matching - exact match."""
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="doc123"
        )
        
        assert rule.matches(AccessScope.DOCUMENT, Permission.READ, "doc123") is True
        assert rule.matches(AccessScope.DOCUMENT, Permission.WRITE, "doc123") is False
        assert rule.matches(AccessScope.DOCUMENT, Permission.READ, "doc456") is False

    def test_access_rule_matches_wildcard(self):
        """Test access rule matching - wildcard."""
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="*"  # Wildcard
        )
        
        assert rule.matches(AccessScope.DOCUMENT, Permission.READ, "doc123") is True
        assert rule.matches(AccessScope.DOCUMENT, Permission.READ, "doc456") is True
        assert rule.matches(AccessScope.DOCUMENT, Permission.WRITE, "doc123") is False

    def test_access_rule_to_dict(self):
        """Test AccessRule serialization."""
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="doc123",
            conditions={"user_id": "user1"}
        )
        
        result = rule.to_dict()
        
        assert result["scope"] == "document"
        assert result["permission"] == "read"
        assert result["resource_id"] == "doc123"
        assert result["conditions"] == {"user_id": "user1"}

    def test_access_rule_from_dict(self):
        """Test AccessRule creation from dictionary."""
        data = {
            "scope": "document",
            "permission": "read",
            "resource_id": "doc123",
            "conditions": {"user_id": "user1"}
        }
        
        rule = AccessRule.from_dict(data)
        
        assert rule.scope == AccessScope.DOCUMENT
        assert rule.permission == Permission.READ
        assert rule.resource_id == "doc123"
        assert rule.conditions == {"user_id": "user1"}


class TestUserContext:
    """Test UserContext class."""
    
    def test_user_context_creation(self):
        """Test UserContext creation."""
        context = UserContext(
            user_id="user123",
            group_ids=["group1", "group2"],
            permissions=[Permission.READ, Permission.WRITE],
            metadata={"role": "analyst"}
        )
        
        assert context.user_id == "user123"
        assert context.group_ids == ["group1", "group2"]
        assert context.permissions == [Permission.READ, Permission.WRITE]
        assert context.metadata == {"role": "analyst"}

    def test_user_context_defaults(self):
        """Test UserContext with default values."""
        context = UserContext(user_id="user123")
        
        assert context.group_ids == []
        assert context.permissions == []
        assert context.metadata == {}
        assert isinstance(context.created_at, datetime)

    def test_user_context_has_permission(self):
        """Test permission checking."""
        context = UserContext(
            user_id="user123",
            permissions=[Permission.READ, Permission.WRITE]
        )
        
        assert context.has_permission(Permission.READ) is True
        assert context.has_permission(Permission.WRITE) is True
        assert context.has_permission(Permission.DELETE) is False
        assert context.has_permission(Permission.ADMIN) is False

    def test_user_context_is_in_group(self):
        """Test group membership checking."""
        context = UserContext(
            user_id="user123",
            group_ids=["group1", "group2", "group3"]
        )
        
        assert context.is_in_group("group1") is True
        assert context.is_in_group("group2") is True
        assert context.is_in_group("group4") is False

    def test_user_context_has_any_group(self):
        """Test checking membership in any of multiple groups."""
        context = UserContext(
            user_id="user123",
            group_ids=["group1", "group2"]
        )
        
        assert context.has_any_group(["group1", "group3"]) is True
        assert context.has_any_group(["group2", "group4"]) is True
        assert context.has_any_group(["group3", "group4"]) is False

    def test_user_context_to_dict(self):
        """Test UserContext serialization."""
        context = UserContext(
            user_id="user123",
            group_ids=["group1", "group2"],
            permissions=[Permission.READ, Permission.WRITE],
            metadata={"role": "analyst"}
        )
        
        result = context.to_dict()
        
        assert result["user_id"] == "user123"
        assert result["group_ids"] == ["group1", "group2"]
        assert result["permissions"] == ["read", "write"]
        assert result["metadata"] == {"role": "analyst"}

    def test_user_context_from_dict(self):
        """Test UserContext creation from dictionary."""
        data = {
            "user_id": "user123",
            "group_ids": ["group1", "group2"],
            "permissions": ["read", "write"],
            "metadata": {"role": "analyst"}
        }
        
        context = UserContext.from_dict(data)
        
        assert context.user_id == "user123"
        assert context.group_ids == ["group1", "group2"]
        assert context.permissions == [Permission.READ, Permission.WRITE]
        assert context.metadata == {"role": "analyst"}


class TestRBACManager:
    """Test RBACManager class."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Milvus client."""
        client = Mock()
        client.alias = "default"
        return client

    @pytest.fixture
    def sample_user_context(self):
        """Sample user context."""
        return UserContext(
            user_id="user123",
            group_ids=["analysts", "researchers"],
            permissions=[Permission.READ, Permission.WRITE]
        )

    def test_rbac_manager_creation(self, mock_client):
        """Test RBACManager creation."""
        manager = RBACManager(mock_client)
        
        assert manager.client == mock_client
        assert manager.enable_rbac is True
        assert len(manager._access_rules) == 0
        assert len(manager._user_contexts) == 0

    def test_rbac_manager_disabled(self, mock_client):
        """Test RBACManager with RBAC disabled."""
        manager = RBACManager(mock_client, enable_rbac=False)
        
        assert manager.enable_rbac is False

    def test_add_user_context(self, mock_client, sample_user_context):
        """Test adding user context."""
        manager = RBACManager(mock_client)
        
        manager.add_user_context(sample_user_context)
        
        assert "user123" in manager._user_contexts
        assert manager._user_contexts["user123"] == sample_user_context

    def test_get_user_context_exists(self, mock_client, sample_user_context):
        """Test getting existing user context."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        context = manager.get_user_context("user123")
        
        assert context == sample_user_context

    def test_get_user_context_not_exists(self, mock_client):
        """Test getting non-existing user context."""
        manager = RBACManager(mock_client)
        
        context = manager.get_user_context("nonexistent")
        
        assert context is None

    def test_remove_user_context(self, mock_client, sample_user_context):
        """Test removing user context."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        assert "user123" in manager._user_contexts
        
        manager.remove_user_context("user123")
        
        assert "user123" not in manager._user_contexts

    def test_add_access_rule(self, mock_client):
        """Test adding access rule."""
        manager = RBACManager(mock_client)
        
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="doc123"
        )
        
        manager.add_access_rule(rule)
        
        assert len(manager._access_rules) == 1
        assert manager._access_rules[0] == rule

    def test_check_permission_allowed(self, mock_client, sample_user_context):
        """Test permission check - allowed."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        # Add rule allowing read access for user
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="doc123",
            conditions={"user_id": "user123"}
        )
        manager.add_access_rule(rule)
        
        result = manager.check_permission(
            user_id="user123",
            permission=Permission.READ,
            resource_id="doc123",
            scope=AccessScope.DOCUMENT
        )
        
        assert result is True

    def test_check_permission_denied(self, mock_client, sample_user_context):
        """Test permission check - denied."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        # No matching access rule
        result = manager.check_permission(
            user_id="user123",
            permission=Permission.READ,
            resource_id="doc123",
            scope=AccessScope.DOCUMENT
        )
        
        assert result is False

    def test_check_permission_rbac_disabled(self, mock_client):
        """Test permission check with RBAC disabled."""
        manager = RBACManager(mock_client, enable_rbac=False)
        
        result = manager.check_permission(
            user_id="user123",
            permission=Permission.READ,
            resource_id="doc123",
            scope=AccessScope.DOCUMENT
        )
        
        # Should allow all when RBAC is disabled
        assert result is True

    def test_check_permission_group_access(self, mock_client, sample_user_context):
        """Test permission check - group access."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        # Add rule allowing read access for group
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.READ,
            resource_id="doc123",
            conditions={"group_id": "analysts"}
        )
        manager.add_access_rule(rule)
        
        result = manager.check_permission(
            user_id="user123",
            permission=Permission.READ,
            resource_id="doc123",
            scope=AccessScope.DOCUMENT
        )
        
        assert result is True

    def test_get_access_filter_user_only(self, mock_client, sample_user_context):
        """Test access filter generation - user only."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        filter_expr = manager.get_access_filter("user123")
        
        assert 'user_id == "user123"' in filter_expr

    def test_get_access_filter_user_and_groups(self, mock_client, sample_user_context):
        """Test access filter generation - user and groups."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        filter_expr = manager.get_access_filter("user123")
        
        assert 'user_id == "user123"' in filter_expr
        assert 'JSON_CONTAINS(group_ids, "analysts")' in filter_expr
        assert 'JSON_CONTAINS(group_ids, "researchers")' in filter_expr
        assert " OR " in filter_expr

    def test_get_access_filter_rbac_disabled(self, mock_client):
        """Test access filter with RBAC disabled."""
        manager = RBACManager(mock_client, enable_rbac=False)
        
        filter_expr = manager.get_access_filter("user123")
        
        assert filter_expr is None

    def test_get_access_filter_user_not_found(self, mock_client):
        """Test access filter for non-existing user."""
        manager = RBACManager(mock_client)
        
        with pytest.raises(RBACError, match="User context not found"):
            manager.get_access_filter("nonexistent")

    def test_validate_access_success(self, mock_client, sample_user_context):
        """Test access validation - success."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        # Add rule allowing write access
        rule = AccessRule(
            scope=AccessScope.DOCUMENT,
            permission=Permission.WRITE,
            resource_id="*",
            conditions={"user_id": "user123"}
        )
        manager.add_access_rule(rule)
        
        # Should not raise
        manager.validate_access(
            user_id="user123",
            permission=Permission.WRITE,
            resource_id="doc123"
        )

    def test_validate_access_denied(self, mock_client, sample_user_context):
        """Test access validation - denied."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        with pytest.raises(PermissionError, match="Access denied"):
            manager.validate_access(
                user_id="user123",
                permission=Permission.DELETE,
                resource_id="doc123"
            )

    def test_get_user_permissions(self, mock_client, sample_user_context):
        """Test getting user permissions."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        permissions = manager.get_user_permissions("user123")
        
        assert Permission.READ in permissions
        assert Permission.WRITE in permissions
        assert Permission.DELETE not in permissions

    def test_get_user_groups(self, mock_client, sample_user_context):
        """Test getting user groups."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        groups = manager.get_user_groups("user123")
        
        assert "analysts" in groups
        assert "researchers" in groups

    def test_add_user_to_group(self, mock_client, sample_user_context):
        """Test adding user to group."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        manager.add_user_to_group("user123", "new_group")
        
        context = manager.get_user_context("user123")
        assert "new_group" in context.group_ids

    def test_remove_user_from_group(self, mock_client, sample_user_context):
        """Test removing user from group."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        manager.remove_user_from_group("user123", "analysts")
        
        context = manager.get_user_context("user123")
        assert "analysts" not in context.group_ids
        assert "researchers" in context.group_ids

    def test_grant_permission(self, mock_client, sample_user_context):
        """Test granting permission to user."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        manager.grant_permission("user123", Permission.DELETE)
        
        context = manager.get_user_context("user123")
        assert Permission.DELETE in context.permissions

    def test_revoke_permission(self, mock_client, sample_user_context):
        """Test revoking permission from user."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        manager.revoke_permission("user123", Permission.READ)
        
        context = manager.get_user_context("user123")
        assert Permission.READ not in context.permissions
        assert Permission.WRITE in context.permissions

    def test_get_rbac_stats(self, mock_client, sample_user_context):
        """Test getting RBAC statistics."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        # Add some rules
        rule1 = AccessRule(AccessScope.DOCUMENT, Permission.READ, "doc1")
        rule2 = AccessRule(AccessScope.COLLECTION, Permission.WRITE, "col1")
        manager.add_access_rule(rule1)
        manager.add_access_rule(rule2)
        
        stats = manager.get_rbac_stats()
        
        assert stats["total_users"] == 1
        assert stats["total_rules"] == 2
        assert stats["enabled"] is True

    def test_clear_access_rules(self, mock_client):
        """Test clearing all access rules."""
        manager = RBACManager(mock_client)
        
        # Add some rules
        rule1 = AccessRule(AccessScope.DOCUMENT, Permission.READ, "doc1")
        rule2 = AccessRule(AccessScope.COLLECTION, Permission.WRITE, "col1")
        manager.add_access_rule(rule1)
        manager.add_access_rule(rule2)
        
        assert len(manager._access_rules) == 2
        
        manager.clear_access_rules()
        
        assert len(manager._access_rules) == 0

    def test_clear_user_contexts(self, mock_client, sample_user_context):
        """Test clearing all user contexts."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        assert len(manager._user_contexts) == 1
        
        manager.clear_user_contexts()
        
        assert len(manager._user_contexts) == 0

    def test_export_rbac_config(self, mock_client, sample_user_context):
        """Test exporting RBAC configuration."""
        manager = RBACManager(mock_client)
        manager.add_user_context(sample_user_context)
        
        rule = AccessRule(AccessScope.DOCUMENT, Permission.READ, "doc1")
        manager.add_access_rule(rule)
        
        config = manager.export_rbac_config()
        
        assert "users" in config
        assert "rules" in config
        assert "settings" in config
        assert len(config["users"]) == 1
        assert len(config["rules"]) == 1

    def test_import_rbac_config(self, mock_client):
        """Test importing RBAC configuration."""
        config = {
            "users": {
                "user123": {
                    "user_id": "user123",
                    "group_ids": ["group1"],
                    "permissions": ["read"],
                    "metadata": {}
                }
            },
            "rules": [
                {
                    "scope": "document",
                    "permission": "read", 
                    "resource_id": "doc1",
                    "conditions": {}
                }
            ],
            "settings": {
                "enable_rbac": True
            }
        }
        
        manager = RBACManager(mock_client)
        manager.import_rbac_config(config)
        
        assert len(manager._user_contexts) == 1
        assert len(manager._access_rules) == 1
        assert manager.enable_rbac is True


class TestCreateRBACManager:
    """Test create_rbac_manager function."""
    
    def test_create_rbac_manager_default(self):
        """Test creating RBAC manager with defaults."""
        mock_client = Mock()
        
        manager = create_rbac_manager(mock_client)
        
        assert isinstance(manager, RBACManager)
        assert manager.client == mock_client
        assert manager.enable_rbac is True

    def test_create_rbac_manager_disabled(self):
        """Test creating RBAC manager disabled."""
        mock_client = Mock()
        
        manager = create_rbac_manager(mock_client, enable_rbac=False)
        
        assert manager.enable_rbac is False
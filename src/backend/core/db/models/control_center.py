"""Persistent state for the user-facing SafeHarness test/config center."""

from datetime import datetime, timezone

from core.db.engine import Base
from sqlalchemy import JSON, TIMESTAMP, Column, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB

JSONType = JSON().with_variant(JSONB(), "postgresql")


class SafeHarnessTestRun(Base):
    __tablename__ = "safeharness_test_runs"

    run_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False)
    tenant_id = Column(String(256), nullable=False)
    scenario_id = Column(String(64), nullable=False)
    status = Column(String(20), nullable=False, default="completed")
    result = Column(JSONType, nullable=False, default=dict)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    __table_args__ = (Index("idx_safeharness_test_runs_user", "user_id", "created_at"),)


class ConfigCenterDraft(Base):
    __tablename__ = "config_center_drafts"

    draft_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False)
    tenant_id = Column(String(256), nullable=False)
    config_key = Column(String(128), nullable=False)
    agent_id = Column(String(64))
    base_version = Column(String(64), nullable=False)
    payload = Column(JSONType, nullable=False, default=dict)
    validation = Column(JSONType, nullable=False, default=dict)
    status = Column(String(24), nullable=False, default="draft")
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    updated_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc).replace(tzinfo=None), onupdate=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    __table_args__ = (Index("idx_config_center_drafts_user", "user_id", "updated_at"),)


class ConfigCenterVersion(Base):
    __tablename__ = "config_center_versions"

    version_id = Column(String(64), primary_key=True)
    user_id = Column(String(64), nullable=False)
    tenant_id = Column(String(256), nullable=False)
    config_key = Column(String(128), nullable=False)
    agent_id = Column(String(64))
    version = Column(String(64), nullable=False)
    payload = Column(JSONType, nullable=False, default=dict)
    source = Column(String(24), nullable=False, default="publish")
    audit_reference = Column(String(128), nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=lambda: datetime.now(timezone.utc).replace(tzinfo=None))

    __table_args__ = (Index("idx_config_center_versions_user", "user_id", "created_at"),)


__all__ = ["SafeHarnessTestRun", "ConfigCenterDraft", "ConfigCenterVersion"]

-- CreateTable
CREATE TABLE "LiteLLM_ConversationSession" (
    "session_id" TEXT NOT NULL,
    "owner" TEXT NOT NULL,
    "version" INTEGER NOT NULL DEFAULT 0,
    "compaction_policy_id" TEXT,
    "last_upstream_at" TIMESTAMP(3) NOT NULL,
    "expires_at" TIMESTAMP(3),
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ConversationSession_pkey" PRIMARY KEY ("session_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_ConversationItem" (
    "session_id" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "seq" INTEGER NOT NULL,
    "content" JSONB NOT NULL,
    "is_output" BOOLEAN NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ConversationItem_pkey" PRIMARY KEY ("session_id","version","seq")
);

-- CreateTable
CREATE TABLE "LiteLLM_ConversationResponse" (
    "response_id" TEXT NOT NULL,
    "session_id" TEXT NOT NULL,
    "version" INTEGER NOT NULL,
    "item_count" INTEGER NOT NULL,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_ConversationResponse_pkey" PRIMARY KEY ("response_id")
);

-- CreateTable
CREATE TABLE "LiteLLM_CompactionPolicy" (
    "policy_id" TEXT NOT NULL,
    "name" TEXT NOT NULL,
    "source" TEXT NOT NULL,
    "instructions" TEXT NOT NULL,
    "settings" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "created_by" TEXT,
    "updated_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "updated_by" TEXT,

    CONSTRAINT "LiteLLM_CompactionPolicy_pkey" PRIMARY KEY ("policy_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_ConversationSession_owner_idx" ON "LiteLLM_ConversationSession"("owner");

-- CreateIndex
CREATE INDEX "LiteLLM_ConversationSession_expires_at_idx" ON "LiteLLM_ConversationSession"("expires_at");

-- CreateIndex
CREATE INDEX "LiteLLM_ConversationResponse_session_id_idx" ON "LiteLLM_ConversationResponse"("session_id");

-- CreateIndex
CREATE UNIQUE INDEX "LiteLLM_CompactionPolicy_name_key" ON "LiteLLM_CompactionPolicy"("name");

-- AddForeignKey
ALTER TABLE "LiteLLM_ConversationSession" ADD CONSTRAINT "LiteLLM_ConversationSession_compaction_policy_id_fkey" FOREIGN KEY ("compaction_policy_id") REFERENCES "LiteLLM_CompactionPolicy"("policy_id") ON DELETE SET NULL ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_ConversationItem" ADD CONSTRAINT "LiteLLM_ConversationItem_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "LiteLLM_ConversationSession"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_ConversationResponse" ADD CONSTRAINT "LiteLLM_ConversationResponse_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "LiteLLM_ConversationSession"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;


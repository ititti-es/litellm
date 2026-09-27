-- AlterTable
ALTER TABLE "LiteLLM_ConversationItem" ADD COLUMN     "search_text" TEXT;

-- CreateTable
CREATE TABLE "LiteLLM_MemorySummaryNode" (
    "node_id" TEXT NOT NULL,
    "parent_id" TEXT,
    "session_id" TEXT NOT NULL,
    "level" TEXT NOT NULL,
    "span_version" INTEGER,
    "span_first_seq" INTEGER,
    "span_last_seq" INTEGER,
    "summary" TEXT NOT NULL,
    "policy_id" TEXT,
    "model" TEXT,
    "tokens" INTEGER,
    "source_versions" JSONB,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "LiteLLM_MemorySummaryNode_pkey" PRIMARY KEY ("node_id")
);

-- CreateIndex
CREATE INDEX "LiteLLM_MemorySummaryNode_session_id_idx" ON "LiteLLM_MemorySummaryNode"("session_id");

-- CreateIndex
CREATE INDEX "LiteLLM_MemorySummaryNode_parent_id_idx" ON "LiteLLM_MemorySummaryNode"("parent_id");

-- AddForeignKey
ALTER TABLE "LiteLLM_MemorySummaryNode" ADD CONSTRAINT "LiteLLM_MemorySummaryNode_session_id_fkey" FOREIGN KEY ("session_id") REFERENCES "LiteLLM_ConversationSession"("session_id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "LiteLLM_MemorySummaryNode" ADD CONSTRAINT "LiteLLM_MemorySummaryNode_parent_id_fkey" FOREIGN KEY ("parent_id") REFERENCES "LiteLLM_MemorySummaryNode"("node_id") ON DELETE CASCADE ON UPDATE CASCADE;


-- Full-text search (tsvector), hand-added: not expressible in the Prisma schema DSL.
-- Expression indexes; queries must use the identical to_tsvector(...) expression to hit them.
CREATE INDEX "LiteLLM_ConversationItem_search_text_gin_idx" ON "LiteLLM_ConversationItem"
  USING GIN (to_tsvector('english', COALESCE("search_text", '')));

CREATE INDEX "LiteLLM_MemorySummaryNode_summary_gin_idx" ON "LiteLLM_MemorySummaryNode"
  USING GIN (to_tsvector('english', "summary"));

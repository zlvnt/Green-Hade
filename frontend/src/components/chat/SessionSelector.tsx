"use client";

import { cn } from "@/lib/utils";

export function SessionSelector({
  sessions,
  current,
  onSelect,
  onCreate,
  onClear,
  onDelete,
}: {
  sessions: string[];
  current: string;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onClear: () => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="w-48 bg-zinc-900 border-r border-zinc-800 flex flex-col">
      <div className="p-3 border-b border-zinc-800">
        <button
          onClick={onCreate}
          className="w-full px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-xs rounded-lg transition-colors"
        >
          + New Session
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
        {sessions.map((id) => (
          <div key={id} className="group flex items-center gap-1">
            <button
              onClick={() => onSelect(id)}
              className={cn(
                "flex-1 text-left px-3 py-2 rounded-lg text-xs truncate transition-colors",
                id === current
                  ? "bg-zinc-800 text-white"
                  : "text-zinc-400 hover:text-white hover:bg-zinc-800/50",
              )}
            >
              {id}
            </button>
            {id !== "default" && (
              <button
                onClick={() => {
                  if (
                    window.confirm(
                      `Hapus session "${id}"?\nMessages di UI dan backend akan dihapus permanen.`,
                    )
                  ) {
                    onDelete(id);
                  }
                }}
                className="px-2 py-2 text-zinc-500 hover:text-red-400 text-base leading-none transition-colors opacity-0 group-hover:opacity-100 focus:opacity-100"
                title="Delete session"
                aria-label={`Delete session ${id}`}
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>
      <div className="p-2 border-t border-zinc-800">
        <button
          onClick={onClear}
          className="w-full px-3 py-1.5 text-xs text-zinc-500 hover:text-red-400 transition-colors"
        >
          Clear Chat
        </button>
      </div>
    </div>
  );
}

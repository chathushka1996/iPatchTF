"use client";

import { useEffect, useRef, useState } from "react";

import { ChatWindow } from "@/components/community/chat-window";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { useWebSocket } from "@/hooks/use-websocket";
import { useChatStore } from "@/stores/chat-store";
import { ChevronLeft, ChevronRight, Hash, Send } from "lucide-react";

export default function ChatPage() {
  const [message, setMessage] = useState("");
  const [usersPanelOpen, setUsersPanelOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const {
    channels,
    activeChannel,
    messages,
    onlineUsers,
    hasNewMessages,
    setActiveChannel,
    sendMessage,
    markMessagesRead,
  } = useChatStore();

  const { isConnected } = useWebSocket("/ws/chat");

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, activeChannel]);

  const handleSend = () => {
    if (!message.trim() || message.length > 500) return;
    sendMessage(message.trim());
    setMessage("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="container mx-auto flex h-[calc(100vh-4rem)] max-w-7xl gap-0 px-4 py-4">
      {/* Channel list */}
      <aside className="hidden w-56 shrink-0 flex-col border-r border-border pr-4 md:flex">
        <h2 className="mb-3 px-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Channels
        </h2>
        <nav className="flex-1 space-y-1 overflow-y-auto">
          {channels.length === 0 ? (
            <Skeleton className="h-8 w-full" />
          ) : (
            channels.map((channel) => (
              <button
                key={channel.id}
                onClick={() => setActiveChannel(channel.id)}
                className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-sm transition-colors ${
                  activeChannel === channel.id
                    ? "bg-indigo-500/10 text-indigo-500"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                }`}
              >
                <Hash className="h-4 w-4 shrink-0" />
                <span className="truncate">{channel.name}</span>
              </button>
            ))
          )}
        </nav>
      </aside>

      {/* Messages */}
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center justify-between border-b border-border pb-3">
          <div className="flex items-center gap-2">
            <Hash className="h-5 w-5 text-muted-foreground" />
            <h1 className="font-semibold">
              {channels.find((c) => c.id === activeChannel)?.name ?? "general"}
            </h1>
            {!isConnected && (
              <span className="text-xs text-amber-500">Reconnecting...</span>
            )}
          </div>
          <Button
            variant="ghost"
            size="icon"
            className="hidden lg:flex"
            onClick={() => setUsersPanelOpen(!usersPanelOpen)}
            aria-label="Toggle online users"
          >
            {usersPanelOpen ? (
              <ChevronRight className="h-4 w-4" />
            ) : (
              <ChevronLeft className="h-4 w-4" />
            )}
          </Button>
        </div>

        <div className="relative flex-1 overflow-y-auto py-4">
          {hasNewMessages && (
            <button
              onClick={markMessagesRead}
              className="absolute bottom-4 left-1/2 z-10 -translate-x-1/2 rounded-full bg-indigo-500 px-4 py-1.5 text-xs font-medium text-white shadow-lg"
            >
              New messages
            </button>
          )}
          <ChatWindow
            messages={messages[activeChannel] ?? []}
            isLoading={!isConnected && !messages[activeChannel]}
          />
          <div ref={messagesEndRef} />
        </div>

        <div className="flex gap-2 border-t border-border pt-3">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message..."
            maxLength={500}
            className="flex-1"
          />
          <Button onClick={handleSend} disabled={!message.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
        <p className="mt-1 text-right text-xs text-muted-foreground">
          {message.length}/500
        </p>
      </div>

      {/* Online users */}
      {usersPanelOpen && (
        <aside className="hidden w-48 shrink-0 flex-col border-l border-border pl-4 lg:flex">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
            Online — {onlineUsers.length}
          </h2>
          <ul className="flex-1 space-y-2 overflow-y-auto">
            {onlineUsers.map((user) => (
              <li
                key={user.id}
                className="flex items-center gap-2 text-sm"
              >
                <span className="h-2 w-2 shrink-0 rounded-full bg-green-500" />
                <span className="truncate">{user.username}</span>
              </li>
            ))}
          </ul>
        </aside>
      )}
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { ArrowDown, Send } from "lucide-react";

import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { formatRelativeDate } from "@/lib/utils";
import { cn } from "@/lib/utils";
import type { ChatMessage } from "@/types/forum";

interface ChatWindowProps {
  messages: ChatMessage[];
  onSend: (body: string) => void;
  isLoading?: boolean;
  className?: string;
}

export function ChatWindow({
  messages,
  onSend,
  isLoading = false,
  className,
}: ChatWindowProps) {
  const [input, setInput] = useState("");
  const [showNewIndicator, setShowNewIndicator] = useState(false);
  const [isAtBottom, setIsAtBottom] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevLengthRef = useRef(messages.length);

  useEffect(() => {
    if (isAtBottom && scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    } else if (messages.length > prevLengthRef.current) {
      setShowNewIndicator(true);
    }
    prevLengthRef.current = messages.length;
  }, [messages, isAtBottom]);

  const handleScroll = () => {
    if (!scrollRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = scrollRef.current;
    const atBottom = scrollHeight - scrollTop - clientHeight < 50;
    setIsAtBottom(atBottom);
    if (atBottom) setShowNewIndicator(false);
  };

  const scrollToBottom = () => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
      setShowNewIndicator(false);
      setIsAtBottom(true);
    }
  };

  const handleSend = () => {
    const trimmed = input.trim();
    if (!trimmed) return;
    onSend(trimmed);
    setInput("");
  };

  return (
    <div className={cn("flex h-full flex-col rounded-lg border", className)}>
      <ScrollArea className="relative flex-1">
        <div
          ref={scrollRef}
          onScroll={handleScroll}
          className="h-[400px] space-y-3 overflow-y-auto p-4"
        >
          {messages.length === 0 ? (
            <p className="text-center text-sm text-muted-foreground">
              No messages yet. Start the conversation!
            </p>
          ) : (
            messages.map((message) => {
              const displayName =
                message.user.display_name || message.user.username;
              return (
                <div key={message.id} className="flex gap-2">
                  <Link href={`/users/${message.user.username}`}>
                    <Avatar className="h-8 w-8">
                      <AvatarImage src={message.user.avatar_url ?? undefined} />
                      <AvatarFallback name={displayName} />
                    </Avatar>
                  </Link>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-baseline gap-2">
                      <span className="text-sm font-medium">{displayName}</span>
                      <span className="text-xs text-muted-foreground">
                        {formatRelativeDate(message.created_at)}
                      </span>
                    </div>
                    <p className="mt-0.5 text-sm break-words">{message.body}</p>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {showNewIndicator && (
          <Button
            size="sm"
            className="absolute bottom-4 left-1/2 -translate-x-1/2"
            onClick={scrollToBottom}
          >
            <ArrowDown className="mr-1 h-4 w-4" />
            New messages
          </Button>
        )}
      </ScrollArea>

      <div className="flex gap-2 border-t p-3">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a message..."
          maxLength={500}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
        />
        <Button
          size="icon"
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          aria-label="Send message"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

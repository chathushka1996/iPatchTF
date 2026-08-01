import { create } from "zustand";

import type { ChatMessage } from "@/types/forum";

export interface ChatChannel {
  id: string;
  name: string;
  description?: string;
}

interface ChatState {
  messages: Record<string, ChatMessage[]>;
  channels: ChatChannel[];
  activeChannel: string | null;
  setChannels: (channels: ChatChannel[]) => void;
  setActiveChannel: (channelId: string | null) => void;
  addMessage: (channelId: string, message: ChatMessage) => void;
  setMessages: (channelId: string, messages: ChatMessage[]) => void;
  clearMessages: (channelId?: string) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: {},
  channels: [],
  activeChannel: null,

  setChannels: (channels) => set({ channels }),

  setActiveChannel: (channelId) => set({ activeChannel: channelId }),

  addMessage: (channelId, message) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [channelId]: [...(state.messages[channelId] ?? []), message],
      },
    })),

  setMessages: (channelId, messages) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [channelId]: messages,
      },
    })),

  clearMessages: (channelId) =>
    set((state) => {
      if (!channelId) {
        return { messages: {} };
      }

      const { [channelId]: _, ...rest } = state.messages;
      return { messages: rest };
    }),
}));

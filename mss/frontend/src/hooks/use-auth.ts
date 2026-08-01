"use client";

import { useCallback, useEffect, useState } from "react";

import { post as apiPost } from "@/lib/api-client";
import { clearTokens, isAuthenticated as checkAuth, setTokens } from "@/lib/auth";
import { useAuthStore } from "@/stores/auth-store";
import type { TokenResponse } from "@/types/api";
import type { User } from "@/types/user";

import { get } from "@/lib/api-client";

interface LoginCredentials {
  email: string;
  password: string;
}

interface RegisterData {
  username: string;
  email: string;
  password: string;
}

export function useAuth() {
  const { user, setUser, setToken, logout: storeLogout } = useAuthStore();
  const [isLoading, setIsLoading] = useState(false);

  const fetchCurrentUser = useCallback(async () => {
    if (!checkAuth()) return null;

    try {
      const currentUser = await get<User>("/v1/users/me");
      setUser(currentUser);
      return currentUser;
    } catch {
      storeLogout();
      return null;
    }
  }, [setUser, storeLogout]);

  useEffect(() => {
    if (checkAuth() && !user) {
      void fetchCurrentUser();
    }
  }, [user, fetchCurrentUser]);

  const login = useCallback(
    async (credentials: LoginCredentials) => {
      setIsLoading(true);
      try {
        const tokens = await apiPost<TokenResponse>(
          "/v1/auth/login",
          credentials,
        );
        setTokens(tokens);
        setToken(tokens.access_token);

        const currentUser = await get<User>("/v1/users/me");
        setUser(currentUser);
        return currentUser;
      } finally {
        setIsLoading(false);
      }
    },
    [setToken, setUser],
  );

  const register = useCallback(
    async (data: RegisterData) => {
      setIsLoading(true);
      try {
        await apiPost<User>("/v1/auth/register", data);
        return login({ email: data.email, password: data.password });
      } finally {
        setIsLoading(false);
      }
    },
    [login],
  );

  const logout = useCallback(async () => {
    setIsLoading(true);
    try {
      await apiPost("/v1/auth/logout").catch(() => undefined);
    } finally {
      clearTokens();
      storeLogout();
      setIsLoading(false);
    }
  }, [storeLogout]);

  return {
    user,
    isAuthenticated: Boolean(user) || checkAuth(),
    isLoading,
    login,
    register,
    logout,
    fetchCurrentUser,
  };
}

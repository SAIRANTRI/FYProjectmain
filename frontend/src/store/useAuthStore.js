import { create } from "zustand";
import axios from "axios";

axios.defaults.withCredentials = true;

const API_BASE = import.meta.env.MODE === "development"
  ? "http://localhost:5000/api"
  : "/api";

export const useAuthStore = create((set, get) => ({
  user: null,
  isAuthenticated: false,
  isLoading: false,
  error: null,
  isCheckingAuth: true,

  register: async ({ username, email, password }) => {
    set({ isLoading: true, error: null });
    try {
      const res = await axios.post(`${API_BASE}/auth/register`, {
        username,
        email,
        password,
      });

      localStorage.setItem('userId', res.data.user.id);
      
      set({
        user: res.data.user,
        isAuthenticated: true,
        isLoading: false,
      });
      return res.data;
    } catch (err) {
      const errorMessage = err.response?.data?.message || "Registration failed";
      set({
        error: errorMessage,
        isLoading: false,
      });
      throw new Error(errorMessage);
    }
  },

  login: async ({ email, password }) => {
    set({ isLoading: true, error: null });
    try {
      const res = await axios.post(`${API_BASE}/auth/login`, {
        email,
        password,
      });

      localStorage.setItem('userId', res.data.user.id);
      
      set({
        user: res.data.user,
        isAuthenticated: true,
        isLoading: false,
      });
      return res.data;
    } catch (err) {
      const errorMessage = err.response?.data?.message || "Login failed";
      set({
        error: errorMessage,
        isLoading: false,
      });
      throw new Error(errorMessage);
    }
  },

  logout: async () => {
    set({ isLoading: true, error: null });
    try {
      await axios.post(`${API_BASE}/auth/logout`);
      localStorage.removeItem('userId');
      set({
        user: null,
        isAuthenticated: false,
        isLoading: false,
      });
    } catch (err) {
      const errorMessage = "Logout failed";
      set({
        error: errorMessage,
        isLoading: false,
      });
      throw new Error(errorMessage);
    }
  },

  checkAuth: async () => {
    set({ isCheckingAuth: true });
    try {
      // First try to get the user profile
      const res = await axios.get(`${API_BASE}/users/profile`);
      
      localStorage.setItem('userId', res.data._id);
      
      set({
        user: {
          id: res.data._id,
          username: res.data.username,
          email: res.data.email,
          createdAt: res.data.createdAt,
        },
        isAuthenticated: true,
        isCheckingAuth: false,
      });
    } catch (error) {
      // If the token is expired, try to refresh it
      try {
        if (error.response?.status === 401) {
          const refreshRes = await axios.post(`${API_BASE}/auth/refresh-token`);
          
          if (refreshRes.data.user) {
            localStorage.setItem('userId', refreshRes.data.user.id);
            
            set({
              user: refreshRes.data.user,
              isAuthenticated: true,
              isCheckingAuth: false,
            });
            return;
          }
        }
      } catch (refreshError) {
        // If refresh fails, clear auth state
        localStorage.removeItem('userId');
        set({
          user: null,
          isAuthenticated: false,
          isCheckingAuth: false,
        });
      }
      
      // If no refresh attempt or it failed
      localStorage.removeItem('userId');
      set({
        user: null,
        isAuthenticated: false,
        isCheckingAuth: false,
      });
    }
  },

  clearError: () => set({ error: null }),
}));
import { create } from "zustand";
import axios from "axios";

axios.defaults.withCredentials = true;

const apiBase =
  import.meta.env.MODE === "development" ? "http://localhost:5000/api" : "/api";

export const useUploadStore = create((set) => ({
  referenceImages: [],
  poolImages: [],
  loading: false,
  error: null,

  // Upload reference images
  uploadReferenceImages: async (files) => {
    set({ loading: true, error: null });
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("file", file));

      const { data } = await axios.post(
        `${apiBase}/upload/reference`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );

      set((state) => ({
        referenceImages: [...state.referenceImages, data.imageUrl],
        loading: false,
      }));
    } catch (error) {
      console.error("Error uploading reference images:", error);
      set({
        error:
          error.response?.data?.message || "Failed to upload reference images",
        loading: false,
      });
    }
  },

  // Upload pool images
  uploadPoolImages: async (files) => {
    set({ loading: true, error: null });
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));

      const { data } = await axios.post(`${apiBase}/upload/pool`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      set((state) => ({
        poolImages: [...state.poolImages, ...data.images],
        loading: false,
      }));
    } catch (error) {
      console.error("Error uploading pool images:", error);
      set({
        error: error.response?.data?.message || "Failed to upload pool images",
        loading: false,
      });
    }
  },

  // Fetch uploaded images
  fetchUploadedImages: async () => {
    set({ loading: true, error: null });
    try {
      const { data } = await axios.get(`${apiBase}/upload/images`);
      set({
        referenceImages: data.referenceImages, // contains _id and imageUrl
        poolImages: data.poolImages,
        loading: false,
      });
    } catch (error) {
      console.error("Error fetching uploaded images:", error);
      set({
        error:
          error.response?.data?.message || "Failed to fetch uploaded images",
        loading: false,
      });
    }
  },
  // Delete image
  deleteImage: async (imageId, type) => {
    set({ loading: true, error: null });
    try {
      await axios.delete(`${apiBase}/upload/delete/${imageId}`);

      if (type === "reference") {
        set((state) => ({
          referenceImages: state.referenceImages.filter(
            (img) => img._id !== imageId
          ),
          loading: false,
        }));
      } else {
        set((state) => ({
          poolImages: state.poolImages.filter((img) => img._id !== imageId),
          loading: false,
        }));
      }
    } catch (error) {
      console.error("Error deleting image:", error);
      set({
        error: error.response?.data?.message || "Failed to delete image",
        loading: false,
      });
    }
  },

  // Clear error
  clearError: () => set({ error: null }),
}));

import { create } from "zustand";
import axios from "axios";

axios.defaults.withCredentials = true;

const apiBase =
  import.meta.env.MODE === "development" ? "http://localhost:5000/api" : "/api";

export const useUploadStore = create((set, get) => ({
  referenceImages: [],
  poolImages: [],
  loading: false,
  error: null,
  uploadProgress: 0,

  // Upload reference images
  uploadReferenceImages: async (files) => {
    set({ loading: true, error: null, uploadProgress: 0 });
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("file", file));

      const { data } = await axios.post(
        `${apiBase}/upload/reference`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            set({ uploadProgress: percentCompleted });
          },
        }
      );

      // Fetch all images after upload to ensure we have the latest data
      await get().fetchUploadedImages();
      
      set({ loading: false, uploadProgress: 0 });
      return data;
    } catch (error) {
      console.error("Error uploading reference images:", error);
      set({
        error:
          error.response?.data?.message || "Failed to upload reference images",
        loading: false,
        uploadProgress: 0,
      });
      throw error;
    }
  },

  // Upload pool images
  uploadPoolImages: async (files) => {
    set({ loading: true, error: null, uploadProgress: 0 });
    try {
      const formData = new FormData();
      files.forEach((file) => formData.append("files", file));

      const { data } = await axios.post(`${apiBase}/upload/pool`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round(
            (progressEvent.loaded * 100) / progressEvent.total
          );
          set({ uploadProgress: percentCompleted });
        },
      });

      // Fetch all images after upload to ensure we have the latest data
      await get().fetchUploadedImages();
      
      set({ loading: false, uploadProgress: 0 });
      return data;
    } catch (error) {
      console.error("Error uploading pool images:", error);
      set({
        error: error.response?.data?.message || "Failed to upload pool images",
        loading: false,
        uploadProgress: 0,
      });
      throw error;
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
      return data;
    } catch (error) {
      console.error("Error fetching uploaded images:", error);
      set({
        error:
          error.response?.data?.message || "Failed to fetch uploaded images",
        loading: false,
      });
      throw error;
    }
  },
  
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
      throw error;
    }
  },
  clearError: () => set({ error: null }),
}));
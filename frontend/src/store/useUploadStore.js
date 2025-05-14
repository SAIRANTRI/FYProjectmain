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

  clearError: () => set({ error: null }),

  processImages: async () => {
    set({ loading: true, error: null });
    try {
      const { data } = await axios.post(`${apiBase}/model/process-images`);
    
      set({
        taskId: data.taskId,
        processingStatus: data.status,
        loading: false,
      });
    
      return data.taskId;
    } catch (error) {
      console.error("Error processing images:", error);
      set({
        error: error.response?.data?.message || "Failed to process images",
        loading: false,
      });
      return null;
    }
  },

  getProcessingResults: async (taskId) => {
    set({ loading: true, error: null });
    try {
      const { data } = await axios.get(`${apiBase}/model/processing-result/${taskId}`);
    
      set({
        processingResults: data.result,
        loading: false,
      });
    
      return data.result;
    } catch (error) {
      console.error("Error getting processing results:", error);
      set({
        error: error.response?.data?.message || "Failed to get processing results",
        loading: false,
      });
      return null;
    }
  },

  downloadResults: (taskId) => {
    const downloadUrl = `${apiBase}/model/download-results/${taskId}`;
    
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.setAttribute('download', 'classified_images.zip');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }
}));

import mongoose from 'mongoose';

// Define the Result Schema
const resultSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User", // References the User model
      required: true,
    },
    referenceImage: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Photo", // Reference to the reference image
      required: true,
    },
    matches: [
      {
        poolImage: {
          type: mongoose.Schema.Types.ObjectId,
          ref: "Photo", // Reference to matched pool images
          required: true,
        },
        confidence: {
          type: Number,
          required: true,
        },
        matchedAt: {
          type: Date,
          default: Date.now,
        },
      },
    ],
    unmatchedImages: [
      {
        poolImage: {
          type: mongoose.Schema.Types.ObjectId,
          ref: "Photo", // Reference to unmatched pool images
          required: true,
        },
        processedAt: {
          type: Date,
          default: Date.now,
        },
      },
    ],
    taskId: {
      type: String,
      required: true,
      unique: true,
    },
    processedAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true, 
  }
);

// Create and export the Result model
const Result = mongoose.model("Result", resultSchema);

export default Result;

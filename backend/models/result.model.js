import mongoose from 'mongoose';

const resultSchema = new mongoose.Schema(
  {
    user: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: true,
    },
    referenceImage: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "Photo", // Reference to the Photo model, not the image itself
      required: true,
    },
    poolImages: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Photo", // Reference to the Photo model
        required: true,
      },
    ],
    matches: [
      {
        poolImage: {
          type: mongoose.Schema.Types.ObjectId,
          ref: "Photo", // Reference to the Photo model
          required: true,
        }, 
        matchedFace: {
          type: String, // Unique identifier for the matched face, like "Face_1"
        },
        confidence: {
          type: Number,
          required: true,
        },
      },
    ],
  },
  {
    timestamps: true, // Track when the result was created/updated
  }
);

const Result = mongoose.model("Result", resultSchema);   

export default Result; 

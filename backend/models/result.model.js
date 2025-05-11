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
      ref: "Photo",
      required: true,
    },
    poolImages: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Photo",
        required: true,
      },
    ],
    matches: [
      {
        poolImage: {
          type: mongoose.Schema.Types.ObjectId,
          ref: "Photo", 
          required: true,
        }, 
        matchedFace: {
          type: String, 
        },
        confidence: {
          type: Number,
          required: true,
        },
      },
    ],
  },
  {
    timestamps: true, 
  }
);

const Result = mongoose.model("Result", resultSchema);   

export default Result; 
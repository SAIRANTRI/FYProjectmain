import mongoose from 'mongoose';

const ClassificationResultSchema = new mongoose.Schema({
  referenceImageId: {
    type: String,
    required: true,
  },
  referenceImageUrl: {
    type: String,
    required: true,
  },
  matchedImages: [{
    imageId: String,
    imageUrl: String,
    confidence: Number,
    processedAt: Date
  }],
  unmatchedImages: [{
    imageId: String,
    imageUrl: String,
    processedAt: Date
  }],
  processedAt: {
    type: Date,
    default: Date.now
  },
  taskId: {
    type: String,
    required: true,
    unique: true
  }
});

const ClassificationResult = mongoose.model('ClassificationResult', ClassificationResultSchema);

export default ClassificationResult;
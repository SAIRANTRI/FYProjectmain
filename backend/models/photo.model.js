import mongoose from 'mongoose';

const photoSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User', 
      required: true,
    },
    publicId: {
      type: String,
      required: true,
    },
    imageUrl: {
      type: String,
      required: true,
    },
    imageType: {
      type: String,
      enum: ['reference', 'pool'], // To differentiate between reference and pool images
      required: true,
    },
    uploadedAt: {
      type: Date,
      default: Date.now,
    },
    referenceId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Photo',  // Link to the reference image, if it's a pool image
      required: function() {
        return this.imageType === 'pool';  // Only required if the imageType is pool
      },
    },
  },
  {
    timestamps: true, // Automatically adds createdAt and updatedAt fields
  }
);

const Photo = mongoose.model('Photo', photoSchema);

export default Photo;
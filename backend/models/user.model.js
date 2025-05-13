import mongoose from 'mongoose';

const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,
    trim: true,
  },
  profilePicUrl: {
    type: String,
    default: "", // Default profile picture URL
  },
  profilePicPublicId: {
    type: String,
    default: "", // Default public ID for the profile picture
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
  },
  password: {
    type: String,
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now,
  }, 
}, {
  timestamps: true,
});

const User = mongoose.model('User', userSchema);

export default User;

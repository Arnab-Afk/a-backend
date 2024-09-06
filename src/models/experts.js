

import mongoose from 'mongoose';
const Schema = mongoose.Schema;



const expertSchema = new Schema({
  name: {
    type: String,
    required: true
  },
  email: {
    type: String,
    required: true,
    unique: true
  },
  password: {
    type: String,
    required: true
  },
  resume_text: {
    type: String,
    required: true
  },
});
    

const Expert = mongoose.model('Expert', expertSchema);

export default Expert;
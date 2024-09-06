//schema for jobpost
import mongoose from "mongoose";
const Schema = mongoose.Schema;

const experienceSchema = new Schema({
    role: String,
    duration: String
}, { _id: false });
const educationSchema = new Schema({
    degree: String,
    location: String,
    university: String,
    yearOfPassing: Number
}, { _id: false });
const jobPostSchema = new Schema({
    title: { type: String, required: true },
    description: { type: String, required: true },
    skills: [{ type: String, required: true }],
    experience: [experienceSchema],
    education: [educationSchema],
    location: { type: String, required: true },
    company: { type: String, required: true },
}, { timestamps: true });

const jobPost= mongoose.model('JobPost', jobPostSchema);
export default jobPost;
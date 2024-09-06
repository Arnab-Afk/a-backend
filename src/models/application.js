import mongoose from "mongoose";
const Schema = mongoose.Schema;

const applicationSchema = new Schema({
    user: {type: mongoose.Types.ObjectId, ref: "User"},
    jobpost: {type: mongoose.Types.ObjectId, ref: "JobPost"},
    candiditate_relevancy: {type: Number, default: 0},
    job_description: {type: String},
    resume: {type: String},
});

const application= mongoose.model('Application', applicationSchema);
export default application;
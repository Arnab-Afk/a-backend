import express from "express";
import application from "../models/application.js";
import user from '../models/user.js';
const router = express.Router();
import jobPost from "../models/jobpost.js";
router.post("/apply", async (req, res) => {

    try {
        console.log(req.body);
        const similarity = req.body.similarity;
        const resume_str = req.body.resume_str;
        
        const u = await user.findOne();
        
        const newApplication = new application({
            jobpost: req.body.job_id,
            user: u.id,
            resume_text: resume_str,
            candiditate_relevancy: similarity
        });
        await newApplication.save();

        // Return the new application
        res.status(201).json(newApplication);
    } catch (error) {
        console.error(error);
        res.status(400).json({ message: error.message });
    }
});
router.get('/', async (req, res) => {
    const applications = await application.find().populate('user').populate('jobpost');
    res.json(applications);
});
export default router;
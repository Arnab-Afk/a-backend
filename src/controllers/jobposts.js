import express from 'express';
import jobpost from '../models/jobpost.js';



const router = express.Router();

router.post('/create-job', async (req, res) => {
    // const newJobPost = new jobpost({
    //     title: req.body.title,
    //     description: req.body.description,
    //     skills: req.body.skills,
    //     experience: req.body.experience,
    //     education: req.body.education,
    //     location: req.body.location,
    //     company: req.body.company
    // });
    try {
        const newJobPost = new jobpost(req.body);
        await newJobPost.save();
        res.status(201).json(newJobPost);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

router.get('/', async (req, res) => {
    const jobs = await jobpost.find();
    res.json(jobs);
});


export default router;
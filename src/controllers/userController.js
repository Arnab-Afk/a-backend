// controllers/userController.js
import express from 'express';
import User from '../models/user.js';
const router = express.Router();
import Expert from '../models/experts.js';
import dotenv from 'dotenv';
dotenv.config();


router.post('/register', async (req, res) => {
    const newUser = new User({
        email: req.body.email,
        password: req.body.password,
    });
    try {
        const user = await newUser.save();
        res.status(201).json(user);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});

//route to create new expert
router.post('/create-expert', async (req, res) => {
    const newExpert = new Expert({
        name: req.body.name,
        email: req.body.email,
        password: req.body.password,
        resume_text:req.body.resume_text,
    });
    try {
        const expert = await newExpert.save();
        res.status(201).json(expert);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
});


//route to get all users
router.get('/experts', async (req, res) => {
    const users = await Expert.find();
    res.json(users);
});
export default router;
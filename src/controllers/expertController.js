import express from "express";
import expert from "../models/experts.js";

const router = express.Router();

router.post("/create-expert", async (req, res) => {
    const newExpert = new expert({
        name: req.body.name,
        email: req.body.email,
        password: req.body.password,
        skills: req.body.skills,
        experience: req.body.experience,
        education: req.body.education,
        location: req.body.location,
    });
    try {
        const newExpert = new expert(req.body);
        await newExpert.save();
        res.status(201).json(newExpert);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
    });


//route to update expert
router.put('/update-expert/:id', async (req, res) => {
    try {
        const relevance = await fetch(`http://localhost:5000/experts/${req.params.id}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(req.body)
        });
        const expert = await expert.findByIdAndUpdate(req.params.id);
        expert.name = req.body.name;
        expert.email = req.body.email;
        expert.password = req.body.password;
        expert.skills = req.body.skills;
        expert.experience = req.body.experience;
        expert.education = req.body.education;
        expert.location = req.body.location;
        await expert.save();
        res.json(expert);
    } catch (error) {
        res.status(400).json({ message: error.message });
    }
}
);



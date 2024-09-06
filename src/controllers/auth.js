import express from "express";
import user from "../models/user.js";
import expert from "../models/experts.js";

const router = express.Router();

router.post("/user/register", async (req, res) => {
    try {
        console.log(req.body);
        const { name, email, password } = req.body;
        const newUser = new user({ name, email, password });
        await newUser.save();
        console.log(newUser);
        res.send(newUser);
    } catch (error) {
        console.error(error);
        res.status(400).send(error);
    }
});

router.post("/user/login", async (req, res) => {
    try {
        const { email, password } = req.body;
        const User = await user.findOne({ email, password });
        
        if (User) {
            res.send(User);
        } else {
            res.status(400).send("Invalid credentials");
        }
    } catch (error) {
        console.log(error);
        res.status(400).send(error);
    }
});

router.post("/expert/register", async (req, res) => {
    try {
        const { email, password } = req.body;
        const newExpert = new expert({ email, password });
        await newExpert.save();
        res.send("Expert registered successfully");
    } catch (error) {
        res.status(400).send(error);
    }
});

router.post("/expert/login", async (req, res) => {
    try {
        const { email, password } = req.body;
        const expert = await expert.findOne({ email, password });
        if (expert) {
            res.send(expert);
        }
        else {
            res.status(400).send("Invalid credentials");
        }
    } catch (error) {
        res.status(400).send(error);
    }
});

export default router;
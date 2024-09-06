//router/router.js
import express from 'express';
import jobposts from '../controllers/jobposts.js';
import auth from '../controllers/auth.js';
import applications from '../controllers/applications.js';
import user from '../controllers/userController.js';

const router = express.Router();

router.use('/job', jobposts);
router.use('/auth', auth);
router.use('/applications', applications);
// router.use('/applications', applications);
router.use('/user', user);
// router.use('/expert', expert);
export default router; 
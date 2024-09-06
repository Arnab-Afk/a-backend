
import express from 'express';
import connectDB from './config/db.js';
import session from 'express-session';
import dotenv from 'dotenv';
dotenv.config();
import cors from 'cors';

import api from './routes/router.js';


const app = express();

app.use(express.json());
app.use(cors());


connectDB();

app.use('/api', api);




const PORT = process.env.PORT || 5000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
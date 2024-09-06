

const mongoose = require('mongoose');
const Schema = mongoose.Schema;

// Define the sub-schemas for nested documents
const skillsSchema = new Schema({
     
    skillName: String,
    proficiency: String
}, { _id: false });

const languagesSchema = new Schema({
    
    languageName: String,
    fluency: String
}, { _id: false });

const educationSchema = new Schema({
    
    degree: String,
    institution: String,
    yearOfPassing: Number
}, { _id: false });

const experienceSchema = new Schema({
    company: String,
    role: String,
    duration: String
}, { _id: false });

const projectsSchema = new Schema({
    // Define the structure as needed
    // Example:
    projectName: String,
    description: String
}, { _id: false });

const certificationsSchema = new Schema({
    // Define the structure as needed
    // Example:
    certificationName: String,
    issuingOrganization: String,
    dateOfIssue: Date
}, { _id: false });

const achievementsSchema = new Schema({
    // Define the structure as needed
    // Example:
    achievementTitle: String,
    description: String
}, { _id: false });

// Define the main schema for the experts collection
const expertSchema = new Schema({
    name: { type: String, required: true },
    email: { type: String, required: true },
    phone: { type: String, required: true },
    description: { type: String, required: true },
    professional_experience_in_years: { type: Number, required: true },
    skills: [skillsSchema],
    languages: [languagesSchema],
    education: [educationSchema],
    experience: [experienceSchema],
    projects: [projectsSchema],
    certifications: [certificationsSchema],
    achievements: [achievementsSchema],
    is_student: { type: Boolean, required: true },
    is_fresher: { type: Boolean, required: true },
    is_experienced: { type: Boolean, required: true },
    resume: { type: String, required: true }
});

// Create the model from the schema
const Expert = mongoose.model('Expert', expertSchema);

module.exports = Expert;
import React, { useEffect, useRef, useState } from 'react';
import { ThemeProvider, withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';

import DataService from "../../services/DataService";
import styles from './styles';
import backgroundSVG from './b1.svg';
import b2SVG from './b2.svg';
import LoadingAnimation from './loading';
import './fonts.css';

const Home = (props) => {
    const { classes } = props;

    console.log("================================== Home ======================================");

    const inputFile = useRef(null);
    const inputText = useRef(null);

    // Component States
    const [video, setImage] = useState(null);
    const [text, setText] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [fileContent, setFileContent] = useState('');
    const [loading, setLoading] = useState(false);

    // Setup Component
    useEffect(() => {

    }, []);

    // Handlers
    const handleVideoUploadClick = () => {
        inputFile.current.click();
    }

    const handleOnChange = (event) => {
        console.log(event.target.files);
       
        setImage(URL.createObjectURL(event.target.files[0]));

        var formData = new FormData();
        formData.append("file", event.target.files[0]);
        DataService.Predict(formData)
            .then(function (response) {
                console.log(response.data);
                setPrediction(response.data);
                setLoading(false);
            })
    }

    // Handlers
    const handleTextUploadClick = () => {
        inputText.current.click();
    }

    const processKeywords = (ks) => {
        let kl = "";
        for (let i = 0; i < ks.length; i++) {
            kl += String(i+1);
            kl += ". ";
            kl += ks[i];
            kl += "\n";
        }
        return kl;
    }

    const renderMultilineText = (te) => {
        const lines = te.split('\n');
        return (
            <div>
                {lines.map((line, index) => (
                    <React.Fragment key={index}>
                        {line}
                        {index !== lines.length - 1 && <br />} {/* Add <br /> after each line except the last one */}
                    </React.Fragment>
                ))}
            </div>
        );
    };

    function stringToList(inputString) {
        // Split the string by commas and use map to trim whitespace from each element
        return inputString.split(',').map(phrase => phrase.trim());
    }

    const handleTextOnChange = (event) => {
        const file = event.target.files[0];

        if (file) {
            const reader = new FileReader();

            reader.onload = (e) => {
                const content = e.target.result;
                setFileContent(content);
            };

            reader.readAsText(file);
        }

        console.log(event.target.files);
        setText(URL.createObjectURL(event.target.files[0]));

        var formData = new FormData();
        formData.append("file", event.target.files[0]);
        DataService.PredictText(formData)
            .then(function (response) {
                console.log(response.data);
                setPrediction(response.data);
                setLoading(false);
            })
    }

    const containerStyle = {
        background: `url(${backgroundSVG})`,
        backgroundSize: 'cover',
        backgroundRepeat: 'no-repeat',
        width: '100%',
        height: '100vh', // Adjust as needed
        // Add other styles as needed
    };

    const containerStyle2 = {
        background: `url(${b2SVG})`,
        backgroundSize: 'cover',
        backgroundRepeat: 'no-repeat',
        width: '100%',
        height: '100vh', // Adjust as needed
    };

    const tStyle = {
        fontFamily: 'RubikBubbles-Regular',
        color: "#000000"
    };

    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <div style={containerStyle}>
                <Container maxwidth='lg' className={classes.container}>
                    <div align="left">
                        <Typography variant="h7" color="primary" style={{fontSize: "4rem", fontFamily:"Oswald"}}>P A V V Y</Typography>
                        <Typography variant="h6">Our cutting-edge AI assistant helps</Typography> 
                        <Typography variant="h6"> you get the most out of school.</Typography>
                    </div>
                </Container>
                <Container maxwidth='lg' className={classes.container}>
                    <div className={classes.textblock} style={{textAlign: 'left', alignItems: 'left'}}>
                        <Typography variant="h6">With our unparalleled generative AI models at your fingertips, identify keywords and 
                        create unique quizzes to test your knowledge from lecture.</Typography>
                            <Typography variant="h6" style={{ color: '#ff8f92' }}>Activate PAVVY by uploading material!</Typography>  
                    </div>
                    <div className={classes.textblock}>
                            <div className={classes.buttonContainer}>
                            <input
                                type="file"
                                accept=".mp3, .mp4"
                                on
                                autocomplete="off"
                                tabindex="-1"
                                className={classes.fileInput}
                                ref={inputFile}
                                onChange={(event) => handleOnChange(event)}
                            />
                                <Button variant="contained" color="primary" size="large" onClick={() => {
                                    handleVideoUploadClick(); setLoading(true);
                                }}><Typography variant="h4">Upload Video</Typography></Button>
                            </div>
                            <div>
                                <input
                                    type="file"
                                    accept=".txt"
                                    on
                                    autocomplete="off"
                                    tabindex="-1"
                                    className={classes.textInput}
                                    ref={inputText}
                                    onChange={(event) => handleTextOnChange(event)}
                                />
                                <Button variant="contained" color="primary" size="large" width="10rem" onClick={() => {
                                    handleTextUploadClick(); setLoading(true);
                                }}><Typography variant="h4">Upload Text</Typography></Button>
                            </div>
                    </div>
                </Container>
                
                <div style={containerStyle2}>
                <Container maxwidth='lg' className={classes.container} height="40vh">
                    <Container maxwidth='lg' className={classes.opContainer}>
                        <Typography variant="h6" color="primary">Keywords</Typography>
                        <div className={classes.textblock}>
                                {loading &&
                                    <LoadingAnimation />
                                }
                            {prediction &&
                                <Typography variant="h4" className={classes.preds}> 
                                    {(prediction.prediction_label.length < 2) &&
                                        <span className={classes.result}>{"Not enough text in transcript."}</span>
                                    }
                                    {prediction.prediction_label.length >= 2 &&
                                        <span className={classes.result}>
                                            {renderMultilineText(processKeywords(stringToList(prediction.keywords)))}
                                        </span>
                                    }
                                </Typography>
                            }
                        </div>
                    </Container>
                    
                    <Container maxwidth='lg' className={classes.opContainer}>
                        <Typography variant="h6" color="primary">Quiz</Typography>
                        <div className={classes.textblock}>
                            {loading &&
                                    <LoadingAnimation />
                            }
                            {prediction &&
                                <Typography variant="h4" className={classes.preds}>
                                    {prediction.prediction_label.length < 1 &&
                                        <span className={classes.safe}>{"Not enough text in transcript."}</span>
                                    }
                                    {prediction.prediction_label.length >= 1 &&
                                        <span className={classes.result}>
                                            {renderMultilineText(prediction.quiz)}
                                        </span>
                                    }
                                </Typography>
                            }
                        </div>
                    </Container> 
                    
                </Container>
                <Container maxwidth='lg' className={classes.container}>
                    <img className={classes.preview} src={video} />
                </Container> 
                    </div>
                </div>
            </main>
        </div>
    );

    // return (
    //     <div className={classes.root}> 
    //         <main className={classes.main}>
    //             <Container maxWidth='lg' className={classes.container}>
    //                 <div className={classes.dropzone} onClick={() => handleVideoUploadClick()}>
    //                     <input
    //                         type="file"
    //                         accept=".mp3, .mp4"
    //                         on
    //                         autocomplete="off"
    //                         tabindex="-1"
    //                         className={classes.fileInput}
    //                         ref={inputFile}
    //                         onChange={(event) => handleOnChange(event)}
    //                     />
    //                     <div><img className={classes.preview} src={video} /></div>
    //                     <div className={classes.help}>Click to upload video.</div>
    //                 </div>
    //                 <div className={classes.dropzone} onClick={() => handleTextUploadClick()}>
    //                     <input
    //                         type="file"
    //                         accept=".txt"
    //                         on
    //                         autocomplete="off"
    //                         tabindex="-1"
    //                         className={classes.fileInput}
    //                         ref={inputText}
    //                         onChange={(event) => handleTextOnChange(event)}
    //                     />
    //                     <div className={classes.help}>Click to upload text.</div>
    //                     <Typography variant="h6" align='center' text-align='center'>
    //                         {fileContent && (
    //                             <div color='black' >
    //                                 <pre>{fileContent.substring(0, 100)}</pre>
    //                             </div>
    //                         )}
    //                     </Typography>
    //                 </div>
    //             </Container>

    //             <Container maxWidth="lg" className={classes.container}>
                    
    //                 <div className={classes.textblock}>
    //                     <div className={classes.textm}>KEYWORDS</div>
    //                     {prediction &&
    //                         <Typography variant="h4" align='center' text-align='center'>
    //                             {prediction.prediction_label.length < 3 &&
    //                                 <span className={classes.safe}>{"Not enough text in transcript."}</span>
    //                             }
    //                             {prediction.prediction_label.length >= 3 &&
    //                                 <span className={classes.result}>{prediction.prediction_label}</span>
    //                             }
    //                         </Typography>
    //                     }
    //                 </div>

    //                 <div className={classes.textblock}>
    //                     <div className={classes.textm}>QUIZ</div>
    //                     {prediction &&
    //                         <Typography variant="h4" align='center' text-align='center'>
    //                             {prediction.prediction_label.length < 3 &&
    //                                 <span className={classes.safe}>{"Not enough text in transcript."}</span>
    //                             }
    //                             {prediction.prediction_label.length >= 3 &&
    //                                 <span className={classes.result}>{prediction.quiz}</span>
    //                             }
    //                         </Typography>
    //                     }
    //                 </div>  
    //             </Container>
    //         </main>
    //     </div>
    // );
};

export default withStyles(styles)(Home);
import React, { useEffect, useRef, useState } from 'react';
import { ThemeProvider, withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';
import Grid from '@material-ui/core/Grid';
import Button from '@material-ui/core/Button';

import DataService from "../../services/DataService";
import styles from './styles';
import backgroundSVG from './b1.svg';
//import '.../fonts/fonts.css';

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
// BACKGROUND IF IT DOESNT WORK CHANGE TODO TODO TODO
    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxwidth='lg' className={classes.container} style={containerStyle}>
                    <div className={classes.titleblock} align="left">
                        <Typography variant="h7" color="primary">P A V V Y</Typography>
                        <Typography variant="h6" width="10rem">Our cutting-edge AI assistant helps</Typography> 
                        <Typography variant="h6"> you get the most out of school.</Typography>
                    </div>
                </Container>
                <Container maxwidth='lg' className={classes.container}>
                    <div className={classes.textblock}>
                        <Typography variant="h6">Pavvy is a great tool, and he wants to help you. </Typography> 
                    </div>
                    <div className={classes.textblock}>
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
                            handleVideoUploadClick();
                        }}><Typography variant="h4">Upload Video</Typography></Button>
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
                        <Button variant="contained" color="primary" size="large" onClick={() => {
                            handleTextUploadClick();
                        }}><Typography variant="h4">Upload Text</Typography></Button>
                    </div>
                </Container>
                <Container maxwidth='lg' className={classes.container} height="40vh">
                    <Container maxwidth='lg' className={classes.opContainer}>
                        <Typography variant="h6" color="primary">Keywords</Typography>
                        <div className={classes.textblock}>
                            {prediction &&
                                <Typography variant="h4" className={classes.preds}>
                                    {prediction.prediction_label.length < 1 &&
                                        <span className={classes.safe}>{"Not enough text in transcript."}</span>
                                    }
                                    {prediction.prediction_label.length >= 1 &&
                                        <span className={classes.result}>
                                            {renderMultilineText(processKeywords(prediction.prediction_label))}
                                        </span>
                                    }
                                </Typography>
                            }
                        </div>
                    </Container>
                    
                    <Container maxwidth='lg' className={classes.opContainer}>
                        <Typography variant="h6" color="primary">Quiz</Typography>
                        <div className={classes.textblock}>
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
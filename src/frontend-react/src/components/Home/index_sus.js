import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Container from '@material-ui/core/Container';
import Typography from '@material-ui/core/Typography';

import DataService from "../../services/DataService";
import styles from './styles';

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
// BACKGROUND IF IT DOESNT WORK CHANGE TODO TODO TODO
    return (
        <div className={classes.root}>
            <main className={classes.main}>
                <Container maxWidth='lg' className={classes.container}>
                    <div className={classes.dropzone} onClick={() => handleVideoUploadClick()}>
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
                        <div><img className={classes.preview} src={video} /></div>
                        <div className={classes.help}>Click to upload video.</div>
                    </div>
                    <div className={classes.dropzone} onClick={() => handleTextUploadClick()}>
                        <input
                            type="file"
                            accept=".txt"
                            on
                            autocomplete="off"
                            tabindex="-1"
                            className={classes.fileInput}
                            ref={inputText}
                            onChange={(event) => handleTextOnChange(event)}
                        />
                        <div className={classes.help}>Click to upload text.</div>
                        <Typography variant="h6" align='center' text-align='center'>
                            {fileContent && (
                                <div color='black' >
                                    <pre>{fileContent.substring(0, 100)}</pre>
                                </div>
                            )}
                        </Typography>
                    </div>
                </Container>

                <Container maxWidth="lg" className={classes.container}>

                    <div className={classes.textblock}>
                        <div className={classes.textm}>KEYWORDS</div>
                        {prediction &&
                            <Typography variant="h4" align='center' text-align='center'>
                                {prediction.prediction_label.length < 3 &&
                                    <span className={classes.safe}>{"Not enough text in transcript."}</span>
                                }
                                {prediction.prediction_label.length >= 3 &&
                                    <span className={classes.result}>{prediction.prediction_label}</span>
                                }
                            </Typography>
                        }
                    </div>

                    <div className={classes.textblock}>
                        <div className={classes.textm}>QUIZ</div>
                        {prediction &&
                            <Typography variant="h4" align='center' text-align='center'>
                                {prediction.prediction_label.length < 3 &&
                                    <span className={classes.safe}>{"Not enough text in transcript."}</span>
                                }
                                {prediction.prediction_label.length >= 3 &&
                                    <span className={classes.result}>{prediction.quiz}</span>
                                }
                            </Typography>
                        }
                    </div>
                </Container>
            </main>
        </div>
    );
};

export default withStyles(styles)(Home);


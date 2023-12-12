import React, { useEffect, useRef, useState } from 'react';
import { withStyles } from '@material-ui/core';
import Typography from '@material-ui/core/Typography';

import styles from './styles';

const Footer = (props) => {
    const { classes } = props;
    const { history } = props;

    console.log("================================== Footer ======================================");

    // Component States

    // Setup Component
    useEffect(() => {

    }, []);

    return (
        <div className={classes.root} backgroundColor="#FFFFFF">
            <Typography align='center'>
                © 2023 PAVVY AI
            </Typography>
            <Typography align='right' margin='10px 10px'>
                PAVVY doesn't rush you. Don't rush PAVVY.       
            </Typography>
            
            

        </div>
        //<div></div>
    );
};

export default withStyles(styles)(Footer);
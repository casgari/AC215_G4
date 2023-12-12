import {
    createTheme,
} from '@material-ui/core/styles';

import '../fonts/fonts.css';

const Theme = createTheme({
    palette: {
        type: 'light',
        primary: {
            // light: will be calculated from palette.primary.main,
            main: '#591DFF',
            // dark: will be calculated from palette.primary.main,
            // contrastText: will be calculated to contrast with palette.primary.main
        },
        secondary: {
            light: '#EC4126',
            main: '#000000',
            // dark: will be calculated from palette.secondary.main,
            contrastText: '#ffffff',
        },
        // error: will use the default color
        info: {
            light: '#6182DF',
            main: '#6182DF',
            // dark: will be calculated from palette.secondary.main,
            contrastText: '#ffffff',
        },
    },
    typography: {
        // fontFamily: 'Rubik, sans-serif',
        useNextVariants: true,
        h6: {
            color: "#000000",
            fontSize: "1.8rem",
            fontWeight: 800,
            fontFamily: "Assistant"
        },
        h4: {
            color: "#FFFFFF",
            fontSize: "1.1rem",
            fontWeight: 800,
            fontFamily: "Assistant"
        },
        
    },
    overrides: {
        MuiOutlinedInput: {
            root: {
                backgroundColor: "#6182DF",
                position: "relative",
                borderRadius: "4px",
            }
        },
    }
}); //        useNextVariants: true,

export default Theme;

//useNextVariants: True
//     h6: {
//     color: "#6182DF",
//     fontSize: "1.1rem",
//     fontFamily: "Roboto, Helvetica, Arial, sans-serif",
//     fontWeight: 800
// },
//     h5: {
//     color: "#6182DF",
//     fontSize: "1.2rem",
//     fontFamily: "Roboto, Helvetica, Arial, sans-serif",
//     fontWeight: 800
// },
//     h4: {
//     color: "#6182DF",
//     fontSize: "1.8rem",
//     fontFamily: "Roboto, Helvetica, Arial, sans-serif",
//     fontWeight: 900
// },
const styles = theme => ({
    root: {
        flexGrow: 1,
        minHeight: "100vh"
    },
    grow: {
        flexGrow: 1,
    },
    main: {
        backgroundColor: '#FFFFFF'
    },
    container: {
        backgroundColor: 'rgba(255, 255, 255, 0.0)',
        paddingTop: "80px",
        paddingBottom: "20px",
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
        textAlign: "center",
        flex:1.0,
        width: "100%",
        height:'80vh',
        margin: "0px, 0px, 0px, 0px",
    },
    dropzone: {
        flex: 0.5,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center",
        margin: "40px",
        borderWidth: "2px",
        borderRadius: "20px",
        borderColor: "#cccccc",
        borderStyle: "dashed",
        backgroundColor: "#F0E9DF",
        outline: "none",
        transition: "border .24s ease-in-out",
        cursor: "pointer",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        minHeight: "400px",
    }, //        backgroundImage: "url('https://storage.googleapis.com/public_colab_images/ai5/mushroom.svg')",
    titleblock: {
        flex:1.0,
        fontSize: "4.1rem",
        display: "flex",
        flexDirection: "column",
        alignItems:"left",
        textAlign: "left",
        margin: "40px",
        color: "#000000",
        backgroundColor: "none",
    },
    textm: {
        flex: 1.0,
    },
    textblock: {
        flex: 0.5,
        display: "grid",
        flexDirection: "column",
        alignItems: "center",
        margin: "40px",
        borderWidth: "2px",
        borderRadius: "20px",
        backgroundColor: 'rgba(255, 255, 255, 0.0)',
        color: "#000000",
        textAlign: "center",
        outline: "none",
        transition: "border .24s ease-in-out",
        cursor: "pointer",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "center",
        minHeight: "400px", 
    },
    fileInput: {
        display: "none",
    },
    textInput: {
        display: "none",
    },
    preview: {
        width: "40%",
    },
    help: {
        color: "#302f2f"
    },
    result: {
        color: "#000000",
        fontSize: "12px",
    },
});

export default styles;
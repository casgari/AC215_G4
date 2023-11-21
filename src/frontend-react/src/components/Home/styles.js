
const styles = theme => ({
    root: {
        flexGrow: 1,
        minHeight: "100vh"
    },
    grow: {
        flexGrow: 1,
    },
    main: {

    },
    container: {
        backgroundColor: "#F0E9DF",
        paddingTop: "80px",
        paddingBottom: "20px",
        display: "flex",
        flexDirection: "row",
        alignItems: "center",
    },
    dropzone: {
        flex: 0.5,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
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
    textm: {
        flex: 0.5,
    },
    textblock: {
        flex: 0.5,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        margin: "40px",
        borderWidth: "2px",
        borderRadius: "20px",
        borderColor: "#cccccc",
        borderStyle: "dashed",
        backgroundColor: "#F0E9DF",
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
    preview: {
        width: "100%",
    },
    help: {
        color: "#302f2f"
    },
    result: {
        color: "#000000",
        fontSize: "12px",
        fontFace: "Roboto"
    },
});

export default styles;
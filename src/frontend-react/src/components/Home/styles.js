
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
        paddingTop: "30px",
        paddingBottom: "20px",
    },
    dropzone: {
        flex: 1,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        margin: "20px",
        borderWidth: "2px",
        borderRadius: "2px",
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
    fileInput: {
        display: "none",
    },
    preview: {
        width: "100%",
    },
    help: {
        color: "#302f2f"
    },
    safe: {
        color: "#31a354",
    },
    poisonous: {
        color: "#de2d26",
    },
});

export default styles;
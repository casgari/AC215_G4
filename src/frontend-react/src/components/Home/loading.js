import React from 'react';
import { CircularProgress } from '@material-ui/core';
import { css } from '@emotion/react';

const override = css`
  display: block;
  margin: 0 auto;
  border-color: red;
`;

const LoadingAnimation = () => {
    return (
        <div className="loading-container">
            <CircularProgress color="secondary" size={50} css={override} />
        </div>
    );
};

export default LoadingAnimation;
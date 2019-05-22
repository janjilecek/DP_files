using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent (typeof (BuddhaController))]
public class Buddha : MonoBehaviour {

    public float moveSpeed = 5;
    BuddhaController controller;
    Vector3 moveVelocity;

    // Use this for initialization
    void Start () {
        controller = GetComponent<BuddhaController>();
        Vector3 moveInput = new Vector3(0, 1, 0);
        moveVelocity = moveInput.normalized * moveSpeed;
    }

    // Update is called once per frame
    void Update () {
        if (Input.GetKey("space"))
        {
            //controller.Move(moveVelocity);
            controller.DoMove();
        }
    }
}

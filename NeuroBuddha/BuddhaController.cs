using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LSL;
using UnityEngine.UI;

public static class ExtensionMethods
{
    public static float Remap(this float value, float from1, float to1, float from2, float to2)
    {
        return (value - from1) / (to1 - from1) * (to2 - from2) + from2;
    }
}

[RequireComponent (typeof (Rigidbody))]
public class BuddhaController : MonoBehaviour {
    public int bestTime;
    public GameObject collid;
    public float nowTime;
    public Text now;
    public Text best;
    Rigidbody rigidBody;
    Vector3 velocity;
    float[] sampleAlpha;
    float[] sampleBeta;
    float[] sampleTheta;
    liblsl.StreamInlet inletAlpha;
    liblsl.StreamInlet inletBeta;
    liblsl.StreamInlet inletTheta;
    public Transform endMarker;
    public Transform startMarker;

    bool lerping = false;
    // Movement speed in units/sec.
    public float speed = 0.001F;

    // Time when the movement started.
    private float startTime;

    // Total distance between the markers.
    private float journeyLength;

    public AudioSource[] audioSources;

    public AudioSource start;
    public AudioSource loop;
    public AudioSource bg;
    public AudioSource thud;

    public bool startPlayed = false;

    public bool useLSL = false;
    int seconds = 0;
    // Use this for initialization
    void Start () {
        bestTime = 5;
        nowTime = 0f;
        
        useLSL = true;
        rigidBody = GetComponent<Rigidbody>();
        endMarker = rigidBody.transform;
        startMarker = rigidBody.transform;
        endMarker = GameObject.FindGameObjectWithTag("Finish").transform;
        audioSources = GetComponents<AudioSource>();
        start = audioSources[1];
        loop = audioSources[2];
        bg = audioSources[0];
        thud = audioSources[3];

        if (useLSL)
        {
            Debug.Log("Started");

            Debug.Log("wait until an EEG stream shows up");
            liblsl.StreamInfo[] resultsAlpha = liblsl.resolve_stream("type", "ALPHA");
            liblsl.StreamInfo[] resultsBeta = liblsl.resolve_stream("type", "BETA");
            liblsl.StreamInfo[] resultsTheta = liblsl.resolve_stream("type", "THETA");

            Debug.Log("open an inlet and print some interesting info about the stream (meta-data, etc.)");
            inletAlpha = new liblsl.StreamInlet(resultsAlpha[0]);
            inletBeta = new liblsl.StreamInlet(resultsBeta[0]);
            inletTheta = new liblsl.StreamInlet(resultsTheta[0]);
            //Debug.Log(inletAlpha.info().as_xml());
            sampleAlpha = new float[2];
            sampleBeta = new float[2];
            sampleTheta = new float[2];
        }
        
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.name == "coolid")
        {
            Debug.Log("spadnul");
            nowTime = 0f;
            thud.Play();
        }
    }

    public void Move(Vector3 _velocity)
    {
        velocity = _velocity;
    }

   

    public void DoMove() {
         // Keep a note of the time the movement started.
        lerping = true;
        startTime = Time.time;

        // Calculate the journey length.
        journeyLength = Vector3.Distance(startMarker.position, endMarker.position);
        startMarker = rigidBody.transform;
    }

    public void FixedUpdate()
    {
        rigidBody.MovePosition(rigidBody.position + velocity * Time.fixedDeltaTime);
        //Debug.Log(rigidBody.position + velocity * Time.fixedDeltaTime);

    }

    // Update is called once per frame
    void Update() {
        //Debug.Log("Update");

        if (useLSL)
        {
            inletAlpha.pull_sample(sampleAlpha);
            inletBeta.pull_sample(sampleBeta);
            inletTheta.pull_sample(sampleTheta);

            //float alphed = ExtensionMethods.Remap(sampleAlpha[0], 8.0f, 15.0f, 0.01f, 1.0f);
            //float beted = ExtensionMethods.Remap(sampleBeta[0], 15.0f, 30.0f, 0.01f, 1.0f);
            //float theted = ExtensionMethods.Remap(sampleTheta[0], 4.0f, 8.0f, 0.01f, 1.0f);

            string s = "";
            /*foreach (float f in sampleAlpha)
            {
                s += f.ToString() + ", ";
            }
            s += ";;; ";
            foreach (float f in sampleBeta)
            {
                s += f.ToString() + ", ";
            }
            s += ";;; ";
            foreach (float f in sampleTheta)
            {
                s += f.ToString() + ", ";
            }

            s += " = " + alphed.ToString() + "/" + beted.ToString() + "/" + theted.ToString();*/

            //s += (theted / alphed).ToString() + ", " + (theted / beted).ToString();
            //float nn1 = theted / alphed;
            //float nn2 = theted / beted;

            float nn1 = sampleTheta[0]/sampleAlpha[0];
            float nn2 = sampleTheta[0]/sampleBeta[0];
            
            //lerping = (nn1 > 1.01 || nn1 < 0.99 || nn2 > 1.01 || nn2 < 0.99); // old with crop
            lerping = (nn2 < 1.0f || nn1 < 4.0f);

            //Debug.Log(s);
        }
        
    


        //Debug.Log(velocity);
        // Distance moved = time * speed.

        if (lerping) {

            nowTime += Time.deltaTime;
            seconds = (int) (nowTime % 60);


            now.text = "NOW: " + seconds.ToString();
            best.text = "BEST TIME " + bestTime.ToString();

            if (seconds > bestTime)
            {
                bestTime = seconds;
            }

            if (! start.isPlaying && !startPlayed ) // if the start sound is not being played
            {
                rigidBody.useGravity = false;
                start.Play();
                startPlayed = true;
                startTime = Time.time;
                //startMarker = rigidBody.transform;
            }

            if (startPlayed && ! loop.isPlaying)
            {
                loop.Play();
            }

            rigidBody.useGravity = false;
            float distCovered = (Time.time - startTime) * speed;

            // Fraction of journey completed = current distance divided by total distance.
            journeyLength = Vector3.Distance(startMarker.position, endMarker.position);
           //Debug.Log(journeyLength);
            float fracJourney = distCovered / journeyLength;

            // Set our position as a fraction of the distance between the markers.
            //startMarker = rigidBody.transform;
            transform.position = Vector3.Lerp(startMarker.position, endMarker.position, fracJourney);    
            if (endMarker.position == rigidBody.transform.position)
            {
                //lerping = false;
            }
        } else
        {
            rigidBody.useGravity = true;
            startTime = Time.time;
            startPlayed = false;
            if (loop.isPlaying)
            {
                loop.Stop();
            }
        }
        
    }
}
